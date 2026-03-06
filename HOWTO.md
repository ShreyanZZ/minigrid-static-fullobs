LV_t1_directnNoslip_debug2.py and LV_t1_directnNoslip.py are final python scripts that were completed. 
All other .ipynb or .py files (except for AbstractionRefinement and feedback_LV) starting with "LV_" were made earlier and the 2 files mentioned above were built on top of those.
Command to run these files is:
"nohup python -u LV_1itrlearning_policy_train.py > out_dirns.txt 2>&1 &" (COMMAND FOR LINUX MACHINE) 

feedback_LV has been built completely but hasn't been changed for using the direct expression of decrease condition.

Scratchpad.py has some prewritten snippets of code to check if some function is working as intended.

"storage" saves trained policies and "certis" saves trained certificates

model.py and certificate.py contain the architectures.

The command to train the policy:
"python -m scripts.train --algo ppo --env MiniGrid-Dynamic-Obstacles-16x16-v0 --model "MODEL_NAME" --seed "SEED NUMBER" --save-interval 10 --frames "NO. OF FRAMES TO TRAIN ON" --log-interval 1"

The command to run rollouts using the trained policy:
"python -m scripts.visualize --env MiniGrid-Dynamic-Obstacles-8x8-v0 --model "MODEL_TO_BE_TESTED" --argmax"

The visualize.py contains a snippet which calculates the number of successes.



