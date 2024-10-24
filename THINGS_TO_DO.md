# Things To Do
1. incorporate wandb
2. change the training code
    - F1 / Acc (instead of loss) on Dev
    - After each epoch
    
3. move trainer functions to utils.
    - make a trainer_selecter function
    - make a trainer func for Mistral / Llama etc.

4. get rid of googl/
5. update CLA so they are specific to Gemma.
6. RUN Expirements. 

7. ** remake the AQA and TQE datasets
    Why? Because...
        - I don't know where the original files are
        - I don't know where my original processing code is
        - This repo needs to be reproducible. 


 #["o_proj", "q_proj", "up_proj", "v_proj", "k_proj", "down_proj", "gate_proj"],