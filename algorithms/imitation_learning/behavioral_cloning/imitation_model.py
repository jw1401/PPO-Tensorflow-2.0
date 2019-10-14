import msvcrt
from environment import EnvInfo


class ImitationModel():

    """
        ##  Imitation Model 

            The User acts via Keys as the model 
        
            Get Key Inputs for DEMONSTRATIONS in DISCRETE Action Space
            Not working for CONTINUOUS Spaces

            @ GetActionLogpValue
            Returns:    Event, dummy, dummy
                        Event:  UP = 1, DOWN = 2, LEFT = 3, RIGHT = 4, ESCAPE = None  
                        otherwise 0  
                                
        ##
    """

    def __init__(self, env_info= EnvInfo, **kwargs):

        if not env_info.is_discrete: raise Exception("Not implemented")
       
    def get_action_logp_value(self, dummy):

        event = 0

        if msvcrt.kbhit():
            key = ord(msvcrt.getch()) 

            if key == 119:              # w key
                event = 5
                
            if key == 115:              # s key
                event = 6

            if key == 224:

                key = ord(msvcrt.getch())

                if key == 72:               # UP
                    event = 1
                
                if key == 80:               # DOWN
                    event = 2
                    
                if key == 75:               # LEFT
                    event = 3
                
                if key == 77:               # RIGHT
                    event = 4

            if key == 27:                   # ESC
                print("ESC")
                event = None

        return event, 1, 1                  # Return 0 as dummys for advatntage calc 
