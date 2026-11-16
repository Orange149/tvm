"""Reuse verified one-port transport with a frozen boundary-mode environment."""
import shlex
from c3_tps_aggregation_transport_v2 import TransportBoard


class CostBoard(TransportBoard):
    def start(self,ordinal,mode):
        spec=self.transport['sessions'][ordinal-1]
        if spec['boundary_mode'] not in ('off','on'):
            raise ValueError('BOUNDARY_MODE_INVALID')
        self._cost_launch_mode=spec['boundary_mode']
        self._cost_launch_injections=0
        try:
            result=super().start(ordinal,mode)
            if self._cost_launch_injections!=1:
                raise ValueError('EXPECTED_ONE_MODE_BOUND_SERVER_LAUNCH')
            result['environment']['C3_TPS_COST_MODE']=spec['boundary_mode']
            return result
        finally:
            self._cost_launch_mode=None

    def ssh(self,command,timeout=30):
        marker='exec taskset -c 3 env '
        if getattr(self,'_cost_launch_mode',None) is not None and marker in command:
            if command.count(marker)!=1:
                raise ValueError('AMBIGUOUS_SERVER_LAUNCH')
            command=command.replace(marker,marker+'C3_TPS_COST_MODE='+shlex.quote(self._cost_launch_mode)+' ',1)
            self._cost_launch_injections+=1
        return super().ssh(command,timeout)
