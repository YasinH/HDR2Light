# HDR2Light

HDR2Light is an image processing toolkit that decomposes lights from HDR images into either dome or area lights with plugin support for various DCCs in order to leverage artistic controls over HDRI lighting.

* Intro video: [https://vimeo.com/406863743](https://vimeo.com/406863743)
* Demo video: [https://vimeo.com/419821083](https://vimeo.com/419821083)

## Supported Applications and Renderers

  - Maya, Arnold
  - Houdini, Mantra

## Dependencies

  - python
  - opencv
  - numpy (generally comes with opencv by default)
  - pathlib
  - Maya or Houdini

## Installation

Add the **src** folder path to `PYTHONPATH` or anywhere that can be found by your application environment of choice

## Try it out

  1. Open Maya or Houdini and create an Arnold or Mantra skydome light, then give it an HDR texture as its color
  2. Select the light (or make any selection that contains any number of such lights)
  3. From Maya script editor or Houdini Python source editor, run the following, and the decomposed lights will get created.
  ```  
  import apps
  apps.main.run(lights_count=4, modes=1, radius=1000, blend=25)
  ```
  `lights_count`: Number of lights to extract. This is limited to maximum number of lights decomposed by the decomposer\
  `modes`: A list or int of 0 or 1 for each extracted lights type. 0 for skydome mode, 1 for area mode\
  `radius`: If the extracted light is an area light, radius indicates how far the light should be from the origin\
  `blend`: The amount of edge blur from key lights to environment

## Current limitations

This tool is currently a proof of concept and hasn't been exhaustively tested. You will likely experience problems, in which case, please let me know directly or by submitting an Issue.

  - Input images that are bigger than 8k get resized to 8k.
  - Area lights intensity in Mantra is not accurate. You need to manually adjust the exposure    
  - Only basic texture connections to lights are supported. Indirect connections will not be recognized
  - The blend amount is quite erroneous, it's better to keep it at the default for now.
