YEAR=${1-2008}

mkdir -p data
wget -P data "http://stat-computing.org/dataexpo/2009/${YEAR}.csv.bz2"
bzip2 -dk "data/${YEAR}.csv.bz2"
