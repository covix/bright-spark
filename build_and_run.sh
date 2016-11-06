INPUT_FILE=$1

mvn package -f bdproj && \
    /opt/spark/bin/spark-submit --class lucnik.App bdproj/target/bdproj-1.0-SNAPSHOT.jar $INPUT_FILE

