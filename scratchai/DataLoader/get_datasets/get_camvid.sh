echo "Downloading the CamVid Dataset"
wget "https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz" --no-check-certificate
echo "CamVid Dataset Successfully fetched!!"

echo "Starting to unzip the fetched files..."
d=$1
if [ -z "$1" ]; then
    d=`pwd`
fi
tar -xf camvid.tgz -C $d
cd "$d/camvid/"
mkdir _images _labels
mv images/* _images/
mv labels/* _labels/
mv _images images/
mv _labels labels/
echo "Fetched files untared successfully!!"
