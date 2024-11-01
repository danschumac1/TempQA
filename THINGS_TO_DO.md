# Things To Do
- Choose generation parameters
    - MenatQA Mixed trained
    - MenatQA Relevant trained
    or
    - MenatQA Mixed trained rel eval
    - MenatQA Mixed trained rand eval
    - MenatQA Mixed trained no eval
    - MenatQA Mixed trained wd eval


- incorporate wandb
    
- move trainer functions to utils.
    - make a trainer func for Mistral / Llama etc.


- **remake the AQA and TQE datasets**
    Why? Because...
        - I don't know where the original files are
        - I don't know where my original processing code is
        - This repo needs to be reproducible. 


 #["o_proj", "q_proj", "up_proj", "v_proj", "k_proj", "down_proj", "gate_proj"],