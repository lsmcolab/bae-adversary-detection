# It Is Among Us: Identifying Adversaries in Ad-hoc Domains Using Q-valued Bayesian Estimations

<i>In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems. 2024.</i> <a href="#alves2024amongus">[1]</a>

Please cite us! 😄🤓

```
@inproceedings{alves2024amongus,
  author = {do Carmo Alves, Matheus Aparecido and Varma, Amokh and Elkhatib, Yehia and Soriano Marcolino, Leandro},
  title = {It Is Among Us: Identifying Adversaries in Ad-hoc Domains Using Q-valued Bayesian Estimations},
  year = {2024},
  isbn = {},
  address = {Auckland, New Zealand},
  abstract = {Ad-hoc teamwork models are crucial for solving distributed tasks in environments with unknown teammates. In order to improve performance, agents may collaborate in the same environment, trusting each other and exchanging information. However, what happens if there is an impostor among us? In this paper, we present BAE, a novel and efficient framework for online planning and estimation within ad-hoc teamwork domains where there is an adversarial agent disguised as teammates. Our approach considers the identification of the impostor through a process we term ``Q-valued Bayesian Estimations''. BAE can identify the adversary at the same time it performs ad-hoc estimation in order to improve coordination. Our results show that BAE has superior accuracy and faster reasoning capabilities in comparison to the state-of-the-art.},
  booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
  numpages = {9},
  series = {AAMAS 2024}
}
```

## WHAT IS BAE? :open_mouth:

<p style="text-align: justify; text-indent: 10px;" >
<i>BAE</i> is a novel and efficient framework for online planning in ad-hoc teamwork domains capable of performing estimations about adversarial agent disguised as a teammate. Our approach considers the identification of the impostor through a process we term ``Q-valued Bayesian Estimations''. BAE can identify the adversary at the same time it performs ad-hoc estimation in order to improve coordination. Our results show that BAE has superior accuracy and faster reasoning capabilities in comparison to the state-of-the-art. More detail and information about our approach can be found in <a href="#alves2024amongus">our paper [1]</a>.
</p>

        
## SUMMARY

In this README you can find:

- [It Is Among Us: Identifying Adversaries in Ad-hoc Domains Using Q-valued Bayesian Estimations](#it-is-among-us-identifying-adversaries-in-ad-hoc-domains-using-q-valued-bayesian-estimations)
  - [WHAT IS BAE? :open\_mouth:](#what-is-bae-open_mouth)
  - [SUMMARY](#summary)
  - [GET STARTED](#get-started)
    - [1. Dependencies :pencil:](#1-dependencies-pencil)
    - [2. Usage :muscle:](#2-usage-muscle)
  - [More details about BAE](#more-details-about-bae)
  - [REFERENCES](#references)

## GET STARTED

### 1. Dependencies :pencil:

<b>- About this repository</b>

This repository represents a streamlined version of the environment used during our research and proposal of BAE.
We removed some files and improved comments in order to facilitate your reading and understanding through the code. :smile:

As mentioned in our paper, we utilized the <a href="#alves2022adleapmas"><i>AdLeap-MAS</i> framework [2]</a> to conduct all experiments and analyze the results. Therefore, the dependencies outlined here mirror those of the framework; however, we provide the minimal set required to run BAE's code, the baselines and the benchmarks presented in the paper, double-check `requirements.txt`.

<b>- Encountering issues while running our code?</b> :fearful:

<p style="text-align: justify; text-indent: 0px;">
 If you find yourself unable to resolve them using our tutorial, we recommend consulting the <a href="https://github.com/lsmcolab/adleap-mas/">AdLeap-MAS GitHub page</a> for additional guidance on troubleshooting common problems or contact us here on GitHub!
</p>

------------------------
### 2. Usage :muscle:

<b>- Quick experience</b>

For a quick experience, we recommend running the default `main.py` file, which will run a BAE's experiment in the Level-based Foraging environment, small scenario. By default, the display will pop-up for visual evaluation of the agent's behaviour and a result file will be created in `results/` folder, which can be directly used in plots later.

<b>- Running different environments and baselines</b>

If you want to run your experiment in other environments, you will find some options at the top of the `main.py` file.

```python
# 1. Setting the environment
method = 'mcts'                 # choose your method (we used only mcts in this paper)
scenario_id = 5                 # define your scenario configuration. Options: [5,6,7,8]
estimation_method = 'bae'       # choosing your estimation method. Options: ['bae','aga','abu','oeata_a]

display = True                  # choosing to turn on or off the display
```

Directly, you can change the Level-based Foraging scenario by modifying the `scenario_id` variable there.
We have 4 different options for `scenarios_id`: `[5,6,7,8]`.
All scenarios were tested and evaluated in our paper.

Now, if you want to change the estimation_method used in the experiments, you can change the `estimation_method` variable.
In summary, we have 4 different estimation methods available: `['bae','aga','abu','oeata_a]`.
All baselines are introduced and tested in our paper.

Finally, you can choose to turn on or off the display using `display = True` or `display = False`, respectively.

And that's it folks. Easy and ready to use. :innocent:

------------------------
## More details about BAE

*~We are working on it and additional details will be available soon~*

------------------------
## REFERENCES

<a name="alves2024amongus">[1]</a> Matheus Aparecido do Carmo Alves, Amokh Varma, Yehia Elkhatib, and Leandro Soriano Marcolino. 2024. <b>It Is Among Us: Identifying Adversaries in Ad-hoc Domains Using Q-valued Bayesian Estimations</b>. In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS '24). Auckland, New Zealand.

<a name="alves2022adleapmas">[2]</a> Matheus Aparecido do Carmo Alves, Amokh Varma, Yehia Elkhatib, and Leandro Soriano Marcolino. 2022. <b>AdLeap-MAS: An Open-source Multi-Agent Simulator for Ad-hoc Reasoning</b>. In Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS '22). International Foundation for Autonomous Agents and Multiagent Systems, Richland, SC, 1893–1895.
