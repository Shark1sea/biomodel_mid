from collections import OrderedDict

from brian2 import *

#=========================================================================================
# 方程组
#=========================================================================================

# sAMPA, x, sNMDA, sGABA 是突触前的导通率变量
# S_AMPA, S_NMDA, S_GABA 是突触后的导通率变量
equations = dict(
    E = '''
    dV/dt         = (-(V - V_L) - Isyn/gE) / tau_m_E : volt (unless refractory)
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : amp
    I_AMPA_ext    = gAMPA_ext_E*sAMPA_ext*(V - V_E) : amp
    I_AMPA        = gAMPA_E*S_AMPA*(V - V_E) : amp
    I_NMDA        = gNMDA_E*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    I_GABA        = gGABA_E*S_GABA*(V - V_I) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsAMPA/dt     = -sAMPA/tauAMPA : 1
    dx/dt         = -x/tau_x : 1
    dsNMDA/dt     = -sNMDA/tauNMDA + alpha*x*(1 - sNMDA) : 1
    S_AMPA : 1
    S_NMDA : 1
    S_GABA : 1
    ''',

    I = '''
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I : volt (unless refractory)
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : amp
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : amp
    I_AMPA        = gAMPA_I*S_AMPA*(V - V_E) : amp
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    I_GABA        = gGABA_I*S_GABA*(V - V_I) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_AMPA: 1
    S_NMDA: 1
    S_GABA: 1
    '''
    )

#=========================================================================================
# 参数
#=========================================================================================

modelparams = dict(
    # 通用 LIF 参数
    V_L    = -70*mV,
    Vth    = -50*mV,
    Vreset = -55*mV,

    # 兴奋性 LIF 参数
    gE        = 25*nS,
    tau_m_E   = 20*ms,
    tau_ref_E = 2*ms,

    # 抑制性 LIF 参数
    gI        = 20*nS,
    tau_m_I   = 10*ms,
    tau_ref_I = 1*ms,

    # 反转电位
    V_E = 0*mV,
    V_I = -70*mV,

    # NMDA 非线性因子
    a = 0.062*mV**-1,
    b = 3.57,

    # 突触时间常数
    tauAMPA = 2*ms,
    tau_x   = 2*ms,
    tauNMDA = 100*ms,
    alpha   = 0.5*kHz,
    tauGABA = 5*ms,
    delay   = 0.5*ms,

    # 外部突触导通率（输入）
    gAMPA_ext_E = 2.1*nS,
    gAMPA_ext_I = 1.62*nS,

    # 未缩放的递归突触导通率（投向兴奋性）
    gAMPA_E = 80*nS,
    gNMDA_E = 264*nS,
    gGABA_E = 520*nS,

    # 未缩放的递归突触导通率（投向抑制性）
    gAMPA_I = 64*nS,
    gNMDA_I = 208*nS,
    gGABA_I = 400*nS,

    # 背景噪声
    nu_ext = 2.4*kHz,

    # 神经元数量
    N_E = 1600,
    N_I = 400,

    # 选择性神经元比例
    fsel = 0.15,

    # Hebb 增强权重
    wp = 1.7
    )

#=========================================================================================
# 模型
#=========================================================================================

class Stimulus(object):
    def __init__(self, Ton, Toff, mu0, coh):
        self.Ton  = Ton
        self.Toff = Toff
        self.mu0  = mu0

        self.set_coh(coh)

    def s1(self, T):
        t_array = np.arange(0, T + defaultclock.dt, defaultclock.dt)
        vals = np.zeros_like(t_array) * Hz
        vals[np.logical_and(self.Ton <= t_array, t_array < self.Toff)] = self.pos
        return TimedArray(vals, defaultclock.dt)

    def s2(self, T):
        t_array = np.arange(0, T + defaultclock.dt, defaultclock.dt)
        vals = np.zeros_like(t_array) * Hz
        vals[np.logical_and(self.Ton <= t_array, t_array < self.Toff)] = self.neg
        return TimedArray(vals, defaultclock.dt)

    def set_coh(self, coh):
        self.pos = self.mu0*(1 + coh/100)
        self.neg = self.mu0*(1 - coh/100)

class Model(object):
    def __init__(self, modelparams, stimulus, T):
        #---------------------------------------------------------------------------------
        # 完成模型规格
        #---------------------------------------------------------------------------------

        # Model parameters
        params = modelparams.copy()

        # 依据神经元数量缩放导通率
        for par in ['gAMPA_E', 'gAMPA_I', 'gNMDA_E', 'gNMDA_I']:
            params[par] /= params['N_E']
        for par in ['gGABA_E', 'gGABA_I']:
            params[par] /= params['N_I']

        # 创建局部变量以便使用
        N_E   = params['N_E']
        fsel  = params['fsel']
        wp    = params['wp']
        delay = params['delay']

        # 子群规模
        N1 = int(fsel*N_E)
        N2 = N1
        N0 = N_E - (N1 + N2)
        params['N0'] = N0
        params['N1'] = N1
        params['N2'] = N2

        # Hebb 减弱权重
        wm = (1 - wp*fsel)/(1 - fsel)
        params['wm'] = wm

        # 群体之间的突触权重矩阵
        self.W = np.asarray([
            [1,  1,  1],
            [wm, wp, wm],
            [wm, wm, wp]
            ])

        #---------------------------------------------------------------------------------
        # 神经元群体
        #---------------------------------------------------------------------------------

        net = OrderedDict() # 网络对象
        exc = OrderedDict() # 兴奋性子群

        # E/I 群体
        for label in ['E', 'I']:
            net[label] = NeuronGroup(params['N_'+label],
                                     equations[label],
                                     method='rk2',
                                     threshold='V > Vth',
                                     reset='V = Vreset',
                                     refractory=params['tau_ref_'+label],
                                     namespace=params)

        # 兴奋性子群划分
        exc[0] = net['E'][:params['N0']]
        exc[1] = net['E'][params['N0']:params['N0'] + params['N1']]
        exc[2] = net['E'][params['N0'] + params['N1']:]

        #---------------------------------------------------------------------------------
        # 背景输入（突触后）
        #---------------------------------------------------------------------------------

        for label in ['E', 'I']:
            net['pg'+label] = PoissonGroup(params['N_'+label], params['nu_ext'])
            net['ic'+label] = Synapses(net['pg'+label], net[label],
                                       on_pre='sAMPA_ext += 1', delay=delay)
            net['ic'+label].connect(condition='i == j')

        #---------------------------------------------------------------------------------
        # 递归输入
        #---------------------------------------------------------------------------------

        # 修改突触前变量（事件触发）
        net['icAMPA'] = Synapses(net['E'], net['E'], on_pre='sAMPA += 1', delay=delay)
        net['icAMPA'].connect(condition='i == j')
        net['icNMDA'] = Synapses(net['E'], net['E'], on_pre='x += 1', delay=delay)
        net['icNMDA'].connect(condition='i == j')
        net['icGABA'] = Synapses(net['I'], net['I'], on_pre='sGABA += 1', delay=delay)
        net['icGABA'].connect(condition='i == j')

        # 将突触前变量映射到突触后变量
        @network_operation(when='start')
        def recurrent_input():
            # AMPA
            S = self.W.dot([sum(self.exc[ind].sAMPA) for ind in range(3)])
            for ind in range(3):
                self.exc[ind].S_AMPA = S[ind]
            self.net['I'].S_AMPA = S[0]

            # NMDA
            S = self.W.dot([sum(self.exc[ind].sNMDA) for ind in range(3)])
            for ind in range(3):
                self.exc[ind].S_NMDA = S[ind]
            self.net['I'].S_NMDA = S[0]

            # GABA
            S = sum(self.net['I'].sGABA)
            self.net['E'].S_GABA = S
            self.net['I'].S_GABA = S

        #---------------------------------------------------------------------------------
        # 外部输入（突触后）
        #---------------------------------------------------------------------------------

        global s1
        s1 = stimulus.s1(T)
        global s2
        s2 = stimulus.s2(T)
        for ind, sname in zip([1, 2], ['s1', 's2']):
            net['pg'+str(ind)] = PoissonGroup(params['N'+str(ind)], '%s(t)' % sname)
            net['ic'+str(ind)] = Synapses(net['pg'+str(ind)], exc[ind],
                                          on_pre='sAMPA_ext += 1', delay=delay)
            net['ic'+str(ind)].connect(condition='i == j')

        #---------------------------------------------------------------------------------
        # 记录脉冲
        #---------------------------------------------------------------------------------

        mons = OrderedDict()
        for label in ['E', 'I']:
            mons['spikes'+label] = SpikeMonitor(net[label], record=True)

        #---------------------------------------------------------------------------------
        # 设置与封装
        #---------------------------------------------------------------------------------

        self.params = params
        self.net    = net
        self.exc    = exc
        self.mons   = mons

        # 将网络对象与监视器加入 NetworkOperation 的 contained_objects
        self.contained_objects = list(self.net.values()) + list(self.mons.values())
        self.contained_objects.extend([recurrent_input])

    def reinit(self):
        # 随机初始化膜电位
        for label in ['E', 'I']:
            self.net[label].V = np.random.uniform(self.params['Vreset'],
                                                  self.params['Vth'],
                                                  size=self.params['N_'+label]) * volt
            
        # 将突触变量置零
        for par in ['sAMPA_ext', 'sAMPA', 'x', 'sNMDA']:
            setattr(self.net['E'], par, 0)
        for par in ['sAMPA_ext', 'sGABA']:
            setattr(self.net['I'], par, 0)

#=========================================================================================
# 仿真
#=========================================================================================

class Simulation(object):
    def __init__(self, modelparams, stimparams, sim_dt, T):
        defaultclock.dt = sim_dt
        self.stimulus = Stimulus(stimparams['Ton'], stimparams['Toff'],
                                 stimparams['mu0'], stimparams['coh'])
        self.model    = Model(modelparams, self.stimulus, T)
        self.network  = Network(self.model.contained_objects)

    def run(self, T, randseed=1):
        # 初始化随机数生成器
        seed(randseed)

        # 初始化并运行
        self.model.reinit()
        self.network.run(T, report='text')

    def savespikes(self, filename_exc, filename_inh):
        print("保存兴奋性脉冲时间到 " + filename_exc)
        data_exc = np.column_stack((self.model.mons['spikesE'].i, self.model.mons['spikesE'].t))
        np.savetxt(filename_exc, data_exc, fmt='%-9d %25.18e',
                   header='{:<8} {:<25}'.format('Neuron', 'Time (s)'))

        print("保存抑制性脉冲时间到 " + filename_inh)
        data_inh = np.column_stack((self.model.mons['spikesI'].i, self.model.mons['spikesI'].t))
        np.savetxt(filename_inh, data_inh, fmt='%-9d %25.18e',
                   header='{:<8} {:<25}'.format('Neuron', 'Time (s)'))

    def loadspikes(self, *args):
        return [np.loadtxt(filename) for filename in args]

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    stimparams = dict(
        Ton  = 0.5*second, # Stimulus onset
        Toff = 1.5*second, # Stimulus offset
        mu0  = 40*Hz,      # Input rate
        coh  = 1.6         # Percent coherence
        )

    sim_dt = 0.02*ms
    T  = 0.1*second

    sim = Simulation(modelparams, stimparams, sim_dt, T)
    sim.run(T, randseed=4)
    sim.savespikes('spikesE.txt', 'spikesI.txt')

    #-------------------------------------------------------------------------------------
    # 光栅图绘制
    #-------------------------------------------------------------------------------------

    # 载入兴奋性脉冲数据
    spikes, = sim.loadspikes('spikesE.txt')

    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(spikes[:,1], spikes[:,0], 'o', ms=2, mfc='k', mew=0)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron index')

    print("保存光栅图到 wang2002.pdf")
    plt.savefig('wang2002.pdf')
