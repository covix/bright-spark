YEAR=${1-2008}
HDFS_DATA_FOLDER="/data"

mkdir -p data
wget -P data "http://stat-computing.org/dataexpo/2009/${YEAR}.csv.bz2"
bzip2 -dk "./data/${YEAR}.csv.bz2"

hdfs dfs -mkdir -p $HDFS_DATA_FOLDER
hdfs dfs -put ./data/${YEAR}.csv $HDFS_DATA_FOLDER
