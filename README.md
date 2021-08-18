# eos-sar

This package provides access to some generic sar processing algorithms. 

Currently, algorithms specific to **Sentinel1** bursts have been implemented. 

Check the usage folder for some examples. 

In order to run the examples in the usage folder, you need to set the follwing 
environment variables: 

* AWS_ACCESS_KEY_ID = "YOUR CREDENTIALS"
* AWS_SECRET_ACCESS_KEY = "YOUR CREDENTIALS"
* AWS_DEFAULT_REGION = kayrros
* AWS_S3_ENDPOINT = s3.kayrros.org

Each file in the usage folder demonstrates a different application: 

* burst_physical_sensor_model.py: Shows the usage of the physical sensor model for projection and localization in a Sentinel-1 burst. 
* burst_registration.py: Shows the registration of a secondary image burst onto a primary image burst. 
* burst_registration_inside.py: Shows the registration of a crop inside a secondary image burst onto a corresponding crop inside a primary image burst. 
* single_image_debursting.py: Shows the debursting of a single primary image. It is possible to define a region of interest (a crop) inside the swath, and only this region will be read and debursted. 
* secondary_image_debursting.py: Shows the debursting of a secondary image after it has been resampled burst by burst onto a primary image. This processing can also be restricted on a region of interest(a crop). 