echo "Downloading the CamVid Dataset"
wget "https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz" --no-check-certificate

echo "CamVid Dataset Successfully fetched!!"

echo "Starting to unzip the fetched files..."
d=pwd
tar -xf camvid.tgz -C $d
echo "Fetched files untared successfully!!"
