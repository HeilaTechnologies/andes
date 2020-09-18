from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.block import Piecewise, Lag, GainLimiter, LagAntiWindupRate, LagAWFreeze
from andes.core.block import PITrackAWFreeze, LagFreeze, DeadBand1, LagRate, PITrackAW
from andes.core.block import LeadLag, Integrator, PIAWHardLimit
from andes.core.var import ExtAlgeb, ExtState, Algeb, AliasState, State

from andes.core.service import ConstService, FlagValue, ExtService, DataSelect, DeviceFinder
from andes.core.service import VarService, ExtendedEvent, Replace, ApplyFunc, VarHold
from andes.core.service import CurrentSign, NumSelect
from andes.core.discrete import Switcher, Limiter, LessThan
from collections import OrderedDict

import numpy as np  # NOQA


class SynchronverterData(ModelData):
    """
    Synchronous inverter model data.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="static generator index",
                            mandatory=True,
                            )
        self.Vn = NumParam(default=480.0,
                           info="AC voltage rating",
                           tex_name='V_n',
                           )
        self.fn = NumParam(default=60.0,
                           info="rated frequency",
                           tex_name='f',
                           )

        self.rsh = NumParam(default=19000, 
                            info="filter resistance", 
                            unit="ohm", z=True,
                            tex_name='r_{sh}'
                            )

        self.xsh = NumParam(default=40000000, 
                            info="filter reactance", 
                            unit="ohm", 
                            z=True,
                            tex_name='x_{sh}'
                            )

        self.p0 = NumParam(default=0.0, 
                            info="AC active power setting", 
                            unit="pu"
                            )
        self.q0 = NumParam(default=0.0, 
                            info="AC reactive power setting", 
                            unit="pu"
                            )
        self.D = NumParam(default=1326.290, 
                          info="Droop coefficient"
                          )

class SynchronverterModel(Model):
    """
    Synchronous inverter implementation.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        #self.group = 'RenGen'
        self.flags.update({'tds': True})


        self.a0 = ExtService(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta_0',
                          )

        self.v0 = ExtService(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V_0',
                          )

        # Initialization of internal power from PV (static generator) instance
 
        self.Pe_0 = ExtService(model='StaticGen',
                             src='p',
                             indexer=self.gen,
                             tex_name='Pe_{0}',
                             )
        self.Qe_0 = ExtService(model='StaticGen',
                             src='q',
                             indexer=self.gen,
                             tex_name='Qe_{0}',
                             )

        self.p0 = ExtService(model='StaticGen',
                             src='p',
                             indexer=self.gen,
                             tex_name='P_0',
                             )
        self.q0 = ExtService(model='StaticGen',
                             src='q',
                             indexer=self.gen,
                             tex_name='Q_0',
                             )

        # --- variables---

        self.paux0 = ConstService(v_str='0',
                                  tex_name='P_{aux0}',
                                  info='const. auxiliary input')

        # self.Ipcmd = Algeb(tex_name='I_{pcmd}',
        #                    info='current component for active power',
        #                    e_str='Pe/v')

        # self.Iqcmd = Algeb(tex_name='I_{qcmd}',
        #                    info='current component for reactive power',
        #                    e_str='Qe/v')

        self.gsh = ConstService(tex_name='g_{sh}', 
                                v_str='re(1/(rsh + 1j * xsh))', 
                                vtype=np.complex
                                )
        self.bsh = ConstService(tex_name='b_{sh}', 
                                v_str='im(1/(rsh + 1j * xsh))', 
                                vtype=np.complex
                                )
        self.es = ConstService(v_str='1.0',
                              tex_name='E_{s}',
                              info='Constant Inverter Voltage'
                              )

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle', 
                          e_str='-(gsh * (es ** 2) - v * es * (gsh * cos(theta_s - a) + bsh * sin(theta_s - a)))'
                        )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V', 
                          info='Bus voltage magnitude',
                          e_str='-(bsh * (es ** 2) + v * es * (gsh * sin(theta_s - a) - bsh * cos(theta_s - a)))'
                        )

        # state variables
        self.theta_s = State(info='inverter angle',
                           unit='rad',
                           v_str='a0',
                           tex_name=r"\theta_s",
                           e_str='(paux0 - gsh * (es ** 2) - v * es * (gsh * cos(theta_s - a) + bsh * sin(theta_s - a)))/D'
                           )

        self.omega = Algeb(info='Inverter frequency',
                          unit='rad/s',
                          v_str='2*pi*fn',
                          tex_name=r'\omega',
                          e_str='-omega + 2*pi*fn + theta_s'
                          )

        def v_numeric(self, **kwargs):
            # disable corresponding `StaticGen`
            self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)

class Synchronverter(SynchronverterData, SynchronverterModel):
    def __init__(self, system, config):
        SynchronverterData.__init__(self)
        SynchronverterModel.__init__(self, system, config)