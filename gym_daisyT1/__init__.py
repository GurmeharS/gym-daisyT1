from gym.envs.registration import register

register(
    id='daisyT1-v0',
    entry_point='gym_daisyT1.envs:DaisyT1Env',
)
register(
    id='daisyT1-extrahard-v0',
    entry_point='gym_daisyT1.envs:DaisyT1ExtraHardEnv',
)

