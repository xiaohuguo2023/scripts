#!/bin/bash

# List of sizes
sizes=(128 256 512 1024 2048 4096 8192)

# Filename to copy
filename="aquavanjaram942_Cijk_Alik_Bjlk_I8II_BH_UserArgs.yaml"

# Loop over each size and copy the file
for size in "${sizes[@]}"; do
  src="../../tt/$size/3_LibraryLogic/$filename"
  dest="./$size/"
  echo "Copying $src to $dest"
  cp "$src" "$dest"
done

