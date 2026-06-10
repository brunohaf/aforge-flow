#!/bin/bash

set -e

./remove_bg_humanoid.sh ./rembg/CoS/humanoids
./remove_bg_creature.sh ./rembg/CoS/misc
./remove_bg_creature.sh ./rembg/items

./remove_bg_humanoid.sh ./rembg/portraits/humanoids
./remove_bg_creature.sh ./rembg/portraits/creatures

./to_webmp.sh ./webmp