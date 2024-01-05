from gymnasium.envs.registration import register

#####
# Major Environment
#####
register(
    id='AdhocReasoningEnv-v1',
    entry_point='src.envs:AdhocReasoningEnv',
)

#####
# Ad-hoc Teamwork Environment
#####
register(
    id='LevelForagingEnv-v2',
    entry_point='src.envs:LevelForagingEnv',
)