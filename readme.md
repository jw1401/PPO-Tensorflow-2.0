# Proximal Policy Optimization (PPO) with Tensorflow 2.0

Deep Reinforcement Learning is a really interesting modern technology and so I decided to implement an PPO (from the family of Policy Gradient Methods) algorithm in Tensorflow 2.0.
Blueprint is the PPO algorithm develped by OpenAI (https://arxiv.org/abs/1707.06347). 

For test reasons I designed four simple training environments with Unity 3D and ML-Agents. You can use this algorithm with executable Unity 3D files and in the Unity 3D Editor.

    - CartPole 
    - RollerBall 
    - BallSorter
    - BallSorterVisualObs

## How to use

1. Clone [PPO Repo](https://github.com/jw1401/PPO-Tensorflow-2.0) and run pip install -e in the PPO folder

2. Clone [Environments Repo](https://github.com/jw1401/Environments)

    Put the repos in an project-folder. You shold have following file structure.

    ```
    Project
        |
        Envs 
        |     
        PPO     
    ```

3. (Optional) If you are familiar with ML-Agents you can also clone this [Repo](https://github.com/jw1401/UnitySDK-MLAgents-Environments) and run from the Unity 3D Editor.

4. Set the configs in the *.yaml file that you want to use

        Standard config = `__Example__.yaml` (is loaded by default if no config is specified) 
        Standard directory = __WORKING_DIRS__/__STANDARD__/__EXAMPLE__.yaml

    - Set **env_name** (path + filename) to the Unity 3D executeable
    - Set **nn_architecure** based on the environment to train (Vec Obs, Visual Obs, mixed, ...)
    - Set **training and policy parameters** (lr, hidden sizes of network, ...)

5. Run python main.py and specify --runner=run-ppo --working_dir=./path/to/your/working_dir --config=your_config.yaml


    ### Run CartPole

        python main.py --runner=run-ppo --working_dir=./__WORKING_DIRS__/CartPole/ --config=CartPole.yaml

    ### Run RollerBall

        python main.py --runner=run-ppo --working_dir=./__WORKING_DIRS__/RollerBall/ --config=RollerBall.yaml
            
    ### Run BallSorter

         python main.py --runner=run-ppo --working_dir=./__WORKING_DIRS__/BallSorter/ --config=BallSorter.yaml

    ### Run BallSorterVisualObs

        python main.py --runner=run-ppo --working_dir=./__WORKING_DIRS__/BallSorterVisualObs/ --config=BallSorterVisualObs.yaml
        
6. Watch the agent learn 

7. Experiment with the environments 
