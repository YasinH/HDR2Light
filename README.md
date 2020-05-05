# HDR2Light

HDR2Light is an image processing toolkit that decomposes lights from HDR images into either dome or areal lights with plugin support for various DCCs

* Demo video: [https://vimeo.com/406863743](https://vimeo.com/406863743)

## Supported Applications and Renderers

  - Houdini, Mantra
  - Maya, Arnold

## Dependencies

  - python
  - opencv
  - numpy (generally comes with opencv by default)
  - pathlib
  - Maya or Houdini

## Installation

Add the *src* folder path to `PYTHONPATH` or anywhere that can be found by your application of choice

## Try it

  1. Open Maya or Houdini and create an Arnold or Mantra skydome light, then give it an HDR texture
  2. Select the light
  3. From Maya script editor or Houdini Python Source Editor, run the following
  ```  
  import apps
  apps.main.run(lights_count=4, modes=1, radius=1000, blend=25)
  ```

## Current limitations

 - If images are
