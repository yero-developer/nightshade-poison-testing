<h1 align="center">nightshade-poison-testing</h1>

<p align="center">
  This project is Yero's testing on the effects of Nightshade poisoning in training diffusion models. 
</p>

## About
Generative AI is becoming more popular and easier to use as technological capabilities increases. These models requires large amount of data, the more clean data the better the model is able to generate the output. There are different types of generative models, the one used in this repository is diffusion. 
<br><br>
"Nightshade is an offensive tool that artists can use as a group to disrupt models that scrape their images without consent (thus protecting all artists against these models)." (Computer Science UChicago., n.d.) This tool is freely available to use by anyone, given with correct supporting platforms (OS and GPU). The paper [<i>Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models</i>](https://arxiv.org/abs/2310.13828) goes into the detail of the Nightshade's effectiviness of poisoning a few generatives model during training. 
<br><br>
Online dicussions on how effective Nightshade poisoning is in the wild are mixed results, with some claiming it works flawlessly and others say it has no impact. With this in mind, I decided to test how effective Nightshade is with a toy u-net diffusion model. 

## Youtube Video
Here is a video showcasing the visualizations of the examples below:
<br>
[Cold Diffusion Variant Models](https://youtu.be/JdRaWuKZdo8)

## Model Setup
- U-Net Diffusion Model
- CIFAR10 and CIFAR100 Datasets
- Linear Beta Scheduler
- DDPM Approach
- 1000 Timesteps

## Datasets Used and Results
| <p align="center">Dataset</p>   | <p align="center">Class Poisoned</p>  | <p align="center">Image Size</p>  | <p align="center">Class Sparsity</p>  | <p align="center">% Class Images Poisoned</p>  | <p align="center">Epoch/Steps Trained To</p>  | <p align="center">Model Disrupted?</p>  |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| <p align="center">CIFAR10</p>   | <p align="center">Dog</p>  | <p align="center">32x32</p>  | <p align="center">10%</p>  | <p align="center">10%</p> | <p align="center">1500/586,891</p> | <p align="center">:x:</p> |
| <p align="center">CIFAR100</p>   | <p align="center">Chair</p>   | <p align="center">32x32</p>  | <p align="center">1%</p>  | <p align="center">10%</p> | <p align="center">375/147,016</p> | <p align="center">:x:</p> |

## Results Explanation
Nightshade poisoning did not disrupt either CIFAR10 or CIFAR100 model training. The loss trends of the normal and poisoned models are similar. The inferenced images of the normal and poisoned models are similar - meaning the poisoned model did not diverage from the learnt noise it predicted similarly in the normal model. Quality of the inferenced images are subpar but does not show the poisoned model inferenced images any transformations into a different class.

## The Train and Validation for CIFAR10
<img src="plots_train_val_epochs/CIFAR10_results.png" alt="cifar10-train-val" width="720" height="480">

## The Train and Validation for CIFAR100
<img src="plots_train_val_epochs/CIFAR100_results.png" alt="cifar100-train-val" width="720" height="480">

## CIFAR10 Dog Inferenced Normal and Posion Comparison
![cifar10-dog-normal-poison](samples_generated_compared/dog_comparison.png)

## CIFAR100 Dog Inferenced Normal and Posion Comparison
![cifar10-chair-normal-poison](samples_generated_compared/chair_comparison.png)

## Possible Issues
Nightshade poisoning relies on high sparisty of a concept in a massive dataset. CIFAR10 and CIFAR100 are both small datasets comprised of 60,000 images (50,000 for training and 10,000 for validation). CIFAR100 has the lowest current used class sparisty of 1%, much lower than the sparsities somewhat reported in the Nightshade paper. Even if 10% of the class images were poisoned, which is fives times more than the trendline of 2% "needed to acheive a 90% attack success" in Figure 13 of the paper, the models trained were not disrupted. A bigger dataset comprising of higher sparsity may be needed to for Nightshade to be effective. But this in turns means there is a minimum level of sparsity is required in a dataset for the training model to be disrupted. This does not take into account in class and weight balancing. In the future, I will test this idea.

The images size could have a role in how effective Nightshade poisoning is. CIFAR10 and CIFAR100 both have RGB 32x32 images. Tiny images compared to what top generative models use for training. 





## How To Run
- To get the required libraries.
```
pip install -r requirements.txt
```


## Instructions to Train




## References
- Computer Science UChicago. (n.d.). What Is Nightshade?. Nightshade. https://nightshade.cs.uchicago.edu/whatis.html
- @misc{shan2024nightshadepromptspecificpoisoningattacks,
      title={Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models}, 
      author={Shawn Shan and Wenxin Ding and Josephine Passananti and Stanley Wu and Haitao Zheng and Ben Y. Zhao},
      year={2024},
      eprint={2310.13828},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2310.13828}, 
}

## Copyright Information
Copyright 2025 Yeshua A. Romero

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
   
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
