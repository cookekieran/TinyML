<#
This script converts an ONNX model to a TensorFlow appropriate
format. This is done as an intermediatary step to prepare a
.tflite file for Edge TPU deployment.

-i: Input path
-o: Output directory folder
#>

onnx2tf -i models/model.onnx -o tf_model

Write-Host "Conversion Complete."