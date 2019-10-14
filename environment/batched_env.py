import os
import cloudpickle
import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from multiprocessing import Process, Queue
from queue import Empty
from . import Env, EnvInfo


def create_batched_env(num_envs, env_params, start= 0):
    """
        Create Batched Environment
        --------------------------
        
            Returns n environments each running in its own process 
            env-fns = create list of create-single-env() with lambda functions | Functions are initialized later in subprocesses
    """
    env_fns = [lambda i=i: create_single_env(env_params, i + start) for i in range(num_envs)]       
    return BatchedEnv(env_fns)


def create_single_env(env_params, idx):
    env = Env(env_params= env_params, worker_id= idx)
    return env


class BatchedEnvBase(ABC):
    def __init__(self):
        pass

    @abstractproperty
    def num_envs(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    def close(self):
        pass


class BatchedEnv(BatchedEnvBase):

    def __init__(self, env_fns):
        super().__init__()

        self.env_info = EnvInfo
        self._procs = []
        self._command_queues = []
        self._result_queues = []
        self._env_fns = env_fns

        for env_fn in env_fns:                  # Initatlize and start worker processes with envs and communication queues
            cmd_queue = Queue()
            res_queue = Queue()
            proc = Process(target=self._worker, args=(cmd_queue, res_queue, cloudpickle.dumps(env_fn)))
            proc.start()
            self._procs.append(proc)
            self._command_queues.append(cmd_queue)
            self._result_queues.append(res_queue)

        value = EnvInfo
        for q in self._result_queues:           # Handshake with procs -> raises Exception if something is bad
            value = self._queue_get(q)
        self.env_info = value                   # Set obs and act space bases on environment

    @property
    def num_envs(self):
        return len(self._procs)

    def reset(self):
        """
            Reset 
            -----
            Returns initial obs, rews, dones after environment reset
        """
        vec_obses = []
        vis_obses = []
        rews = []
        dones = []

        for q in self._command_queues:
            q.put(('reset', None))

        for q in self._result_queues:

            vec_obs, vis_obs, rew, done, info = self._queue_get(q)
            
            vec_obses.append(vec_obs[0]) if self.env_info.is_vector else None
            vis_obses.append(vis_obs[0]) if self.env_info.is_visual else None
            rews.append(rew[0])
            dones.append(done[0])

        vec_obses = np.array(vec_obses, dtype=np.float32) if self.env_info.is_vector else None
        vis_obses = np.array(vis_obses, dtype=np.float32) if self.env_info.is_visual else None

        return vec_obses, vis_obses, np.array(rews, dtype=np.float32), np.array(dones, dtype=np.bool)

    def step(self, actions):
        """
            STEP
            ----
            Step all environments with a list of actions for each environment

            Returns numpy-arrays [vec-obs, vis-obs, rews, dones and infos] with data for each environment
        """

        vec_obses = []
        vis_obses = []
        rews = []
        dones = []
        infos = []

        for q, action in zip(self._command_queues, actions):
            q.put(('step', action))

        for i, q in enumerate(self._result_queues.copy()):

            try:
                vec_obs, vis_obs, rew, done, info = self._queue_get(q)

            except Empty:
                pass
                # sys.stderr.write('restarting worker %d due to hang.\n' % i)
                # self._restart_worker(i)
                # q = self._result_queues[i]
                # self._command_queues[i].put(('reset', None))
                # self._queue_get(q)
                # self._command_queues[i].put(('step', actions[i]))
                # obs, rew, done, info = self._queue_get(q)
                # done = True

            vec_obses.append(vec_obs[0]) if self.env_info.is_vector else None
            vis_obses.append(vis_obs[0]) if self.env_info.is_visual else None
            rews.append(rew[0])
            dones.append(done[0])
            infos.append(info)

        vec_obses = np.array(vec_obses, dtype=np.float32) if self.env_info.is_vector else None
        vis_obses = np.array(vis_obses, dtype=np.float32) if self.env_info.is_visual else None

        return vec_obses, vis_obses, np.array(rews, dtype=np.float32), np.array(dones, dtype=np.bool), infos

    def close(self):
        """
            Close all environments
            ----------------------
        """
        for q in self._command_queues:
            q.put(('close', None))
        for proc in self._procs:
            proc.join()

    def _restart_worker(self, idx):
        pass
        #     os.system('kill -9 $(ps -o pid= --ppid %d)' % self._procs[idx].pid)
        #     self._procs[idx].terminate()
        #     self._procs[idx].join()
        #     cmd_queue = Queue()
        #     res_queue = Queue()
        #     proc = Process(target=self._worker,args=(cmd_queue, res_queue, cloudpickle.dumps(self._env_fns[idx]),))
        #     proc.start()
        #     self._procs[idx] = proc
        #     self._command_queues[idx] = cmd_queue
        #     self._result_queues[idx] = res_queue
        #     self._queue_get(res_queue)

    @staticmethod
    def _worker(cmd_queue, res_queue, env_str):
        """
            Worker for each Environment
            ---------------------------
            Implements a loop for waiting and executing commands from the cmd-queue

            Returns the result from reset or step in environment
        """
        try:
            env = cloudpickle.loads(env_str)()      # Get lambda create_single_env function and execute
            res_queue.put((env.env_info, None))      # Handshake -> return obs and act space

            ep_rew = 0
            ep_len = 0
            info = None

            try:
                while True:                         # Loop waiting and executing commands

                    cmd, arg = cmd_queue.get()

                    if cmd == 'reset':

                        ep_rew = 0
                        ep_len = 0
                        info = None

                        vec_obs, vis_obs, rew, done = env.reset()
                        res_queue.put(((vec_obs, vis_obs, rew, done, info), None))

                    elif cmd == 'step':

                        vec_obs, vis_obs, rew, done = env.step(arg)

                        ep_rew += rew[0]
                        ep_len += 1

                        if done[0]:

                            info = {'ep_rew': ep_rew, 'ep_len': ep_len}
                            ep_rew, ep_len = 0, 0
                            vec_obs, vis_obs, _, _ = env.reset()

                        res_queue.put(((vec_obs, vis_obs, rew, done, info), None))
                        info = None

                    elif cmd == 'close':
                        return

            finally:
                env.close()

        except Exception as exception:      
            res_queue.put((None, exception))        # handle timeout error and hanging envs

    @staticmethod
    def _queue_get(queue=Queue):
        """
        Checks if there is an exception else returns value
        """
        value, exception = queue.get(timeout=20)
        if exception is not None:
            raise exception
        return value
