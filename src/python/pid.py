'''
PID Controller class
'''


class pid:
    def __init__(self, KP, KI, KD, DT, d_f=0):
        self.kp = KP
        self.ki = KI
        self.kd = KD
        self.ei = 0
        self.ep = 0
        self.dt = DT
        if d_f == 0:
            self.filterMode = False
        else:
            self.filterMode = True
            self.lpf = lpf(d_f)
    '''
    Takes an input error signal and returns a control effort
    '''

    def control(self, error):
        self.ei = self.ei + self.dt * error  # forward euler integration
        ed = (error - self.ep)
        self.ep = error
        if self.filterMode:
            ed = self.lpf.stateUpdate(ed)
            return self.kp*error + self.ki * self.ei + self.kd * ed
        else:
            return self.kp*error + self.ki * self.ei + self.kd * ed


class lpf:
    def __init__(self, steps):
        self.alpha = 1 - 1/steps
        self.state = 0

    def stateUpdate(self, x):
        self.state = (1-self.alpha) * x + self.alpha * self.state
        return self.state
