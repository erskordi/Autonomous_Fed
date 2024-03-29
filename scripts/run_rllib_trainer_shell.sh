#!/bin/bash

# Run apt-get clean
echo "Cleaning up package cache"
sudo apt-get clean

# Enter password for sudo
echo "Enter password to continue:"
read -s I8Dcy32X&

# Run rllib trainer
echo "Running rllib trainer"
python rllib_trainer.py --stop-iters 500 --omega-pi 0.5 --omega-psi 0.5 --simulator RF --action-specifications ir_omega_equals --use-penalty True

# Exit
echo "Exiting"