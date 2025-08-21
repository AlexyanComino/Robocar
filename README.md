# Robocar 🚗🤖

## Overview  
Robocar is an autonomous RC car project where we replaced the traditional remote control with Artificial Intelligence.  
Using computer vision and deep learning, the car can perceive its environment and make driving decisions in real time.  

<img width="1980" height="1440" alt="image" src="https://github.com/user-attachments/assets/7029e132-a2a7-4252-bfe5-49fa19a6605c" />

## Objective  
Design and implement a fully autonomous driving system on a small-scale RC car by leveraging AI models running on embedded hardware.  

## Hardware  
- **RC Car**: Traxxas (modified, custom painted)  
- **Onboard Computer**: NVIDIA Jetson Nano  
- **Sensors**: Front-facing camera  
- **Custom Mounts**: Designed and 3D-printed parts to properly install and secure the Jetson Nano onboard  

## How it works  
The car follows a perception → decision → action pipeline:  

1. **Camera** captures the environment in real time.  
2. **Perception AI (U-Net)** generates a road mask and detects lane markings.  
3. **Decision AI** interprets the mask and predicts steering angle and speed.  
4. **Control System** sends commands to the RC car’s motors and wheels.

### System Diagram
<img width="1716" height="270" alt="image" src="https://github.com/user-attachments/assets/19652307-f0b8-4a08-adac-fc8f8169c5c9" />
  
## Training Data  
- **Unity-based simulator** (custom modified for our needs)  
- **Real-world camera captures** from the car  
- **External datasets** for lane detection and driving tasks  

## Results  
➡️ *[Insert GIF or screenshot of the car driving autonomously]*  
➡️ *[Insert sample outputs: lane mask vs camera feed]*  

## Team  
- Alexyan Comino
- Matys Laguerre
- Ambre Cornejo
