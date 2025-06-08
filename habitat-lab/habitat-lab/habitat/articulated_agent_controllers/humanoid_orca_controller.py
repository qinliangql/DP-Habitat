from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    ORCA,
)

from habitat.articulated_agent_controllers.orca import FullState, ObservableState
import numpy as np
import random

class BaseConfig(object):
    def __init__(self):
        pass

# JointState has 2 attributes:
# self.self_state is a FullState
# self.human_states is a list of ObservableStates
class JointState(object):
    # self_state: list of length 9
    # human_states: list of length human_num*5 or nested list [human_num, 5]
    def __init__(self, self_state, human_states):
        assert len(self_state) == 9
        human_states_namedtuple = []
        # if human states is a nested list [human_num, 5]
        if len(np.shape(human_states)) == 2:
            for human_state in human_states:
                assert len(human_state) == 5
                human_states_namedtuple.append(ObservableState(*human_state))
        # if human states is a flatten list of length human_num*5
        else:
            assert len(human_states) % 5 == 0
            human_num = len(human_states) // 5
            for i in range(human_num):
                human_states_namedtuple.append(ObservableState(*human_states[int(i*5):(int((i+1)*5))]))

        self.self_state = FullState(*self_state)
        self.human_states = human_states_namedtuple

class Config(object):
    
    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.05
    orca.time_horizon = 5
    orca.time_horizon_obst = 5
    
    # human config
    humans = BaseConfig()
    humans.visible = True
    # FOV = this values * PI
    humans.FOV = 2.
    
    # robot config
    robot = BaseConfig()
    # whether robot is visible to humans (whether humans respond to the robot's motion)
    robot.visible = True
    
class Humanoid_OCRA:
    def __init__(self,obstacles) -> None:
        self.config = Config()
        self.obstacles = obstacles
        self.policy = ORCA(config=self.config,obstacles=self.obstacles)
        

    
    def get_huamn_actions(self,human_list,robot_state=None,robot_vis=1.0):
        human_actions = []
        for human in human_list:
            if human.walk_flag:
                num = random.random()   # 对机器人做反应的概率
                ob = []
                for other_human in human_list:
                    if other_human != human:
                        ob.append(other_human.get_state(is_full=False))
                if robot_state is not None:
                    if num < robot_vis:
                        ob.append(robot_state)
                    else:
                        if len(ob) > 0:
                            ob.append(ob[-1])
                        else:
                            # print("ob is null here")
                            pass
                    
                state = JointState(human.get_state(is_full=True),ob)
                action = self.policy.predict(state)
            else:
                action = None
            human_actions.append(action)
        return human_actions