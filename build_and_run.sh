INPUT_FILE=${1:-"data/2008_1000.csv"}

mvn package -f bdproj/pom.xml && \
    /opt/spark/bin/spark-submit --class lucnik.App bdproj/target/bdproj-1.0-SNAPSHOT.jar $INPUT_FILE

