from setuptools import setup

#####
# Major Environment
#####
setup(name='AdhocReasoningEnv',
      version='1.0.0',
      install_requires=['gym']
)

#####
# Ad-hoc Teamwork Environment
#####
setup(name='LevelForagingEnv',
      version='2.0.0',
      install_requires=['gym','numpy']
)