#!/bin/bash
# This is the installation script

docker build --tag 'license-classifier-image' .

docker create --name license-classifier-container license-classifier-image
