class DispatchConfig:
    def __init__(self, **kwargs) -> None:
        default = {
            "frontal_opt" : None,
            "optim_horizon" : 96,
            "fcr_commit" : 15,
            "time_mesh" : 15,
            "announced_capacity" : {
                "up" : [20.0 for _ in range (96)],
                "down" : [20.0 for _ in range (96)]
            },
            "cost_of_electricity" : [0.1 - 0.001/(t+1) for t in range(69)],
            "penalties" : {
                "fcr_up" : 1,
                "fcr_down" : 1,
                "SOC_fin" : 0.1
            }
        }

        for k, v in default.items():
            if type(v) == dict:
                for k_, v_ in v.items():
                    setattr(self, f"{k}_{k_}".lower(), kwargs.get(k, v).get(k_, v[k_]))
            setattr(self, k.lower(), kwargs.get(k, v))

        self.evses = []
        for evse_data in kwargs.get("evses", []):
            self.evses.append(EVSE(self, **evse_data))

        assert self.optim_horizon <= min(len(self.announced_capacity_up), len(self.announced_capacity_down))
        assert self.optim_horizon <= len(self.cost_of_electricity)


class EVSE:
    def __init__(self, dispatch_config: DispatchConfig, **kwargs) -> None:
        default = {
            "charging_horizon" : 0,
            "p_charge_max" : 0,
            "p_charge_min" : 0,
            "p_discharge_max" : 0,
            "p_discharge_min" : 0,
            "conversion_rate" : 1,
            "capacity" : 0,
            "SOC_max" : 1.0,
            "SOC_init" : 0.0,
            "SOC_min" : 0.0,
            "SOC_final" : 1.0,
        }
        for k, v in default.items():
            setattr(self, k.lower(), kwargs.get(k, v))


        self.id = kwargs["id"]
        self.soc_final = min(self.soc_final, self.soc_max)
        self.charging_horizon = min(self.charging_horizon, dispatch_config.optim_horizon)
        self.soc_lance_curve = [self.soc_final for _ in range(dispatch_config.optim_horizon + 1)]

        for t in range(self.charging_horizon):
            t += 1
            self.soc_lance_curve[self.charging_horizon - t] = max(self.soc_min,
                self.soc_final - self.p_charge_max * dispatch_config.time_mesh / 60
                * (t - 1) / self.capacity)

        #print(f'lance curve : {self.soc_lance_curve}')
        assert self.soc_lance_curve[0] <= self.soc_init, f"Infeasible SOC init for {self.id}"
