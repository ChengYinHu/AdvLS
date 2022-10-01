# AdvLS
Code and model for "Adversarial Laser Spot: Robust and Covert Physical-World Attack to DNNs" (ACML 2022)
<p align='center'>
  <img src='1.jpg' >
</p>

## Introduction
In the physical world, especially in busy cities, there are so many light resources that they tend to scatter on traffic signs, causing humans to instinctively ignore them. If an attacker deliberately creates an adversarial laser spot that can attack self-driving car systems while lowering human vigilance, it could disrupt traffic and even cause disaster.
In this work, we propose an attack method called adversarial laser spot(AdvLS).which achieves robust and covert physical adversarial attacks to DNNs.

## Requirements
* python == 3.8
* torch == 1.8.0

## Basic Usage
```sh
python digital_test.py --model resnet50 --dataset your_dataset
```

