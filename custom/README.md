# IntrinsicCompositing Extension - CMPT 461

## setup
- follow the original setup instructions for the intrinsicCompositing repo
- activate the virtual environment venv

## run manual pipeline
- firstly, find a foreground and background image you'd like to be composited. The foreground image should be 4 channels, with it's mask as the alpha component.
- next, input the variables at the bottom of the file & decide on the name for the folder that you want all the files to be generated into. These files will be used by the shadow generation pipeline to create shadows.
- Run `python manual_pipeline.py`

## generate shadow & final result
- first, input the variables at the bottom of the file & select the folder you wish to generate a shadow for. You can decide on the blur amount to
- Run `python manual_pipeline.py`
- You result will be written to the folder from above
