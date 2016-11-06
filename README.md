# bright-spark
Repo for BigData project

## Download dataset
Run `./download [year]`

## Create App, compile and stuff:
### Create project
```
mvn archetype:generate
```
filter by *scala* and select something like `net.alchim31.maven:scala-archetype-simple`

### Create a Spark application
Add the following dependecy to `pom.xml`

```
<dependency>
  <groupId>org.apache.spark</groupId>
  <artifactId>spark-core_2.11</artifactId>
  <version>2.0.0</version>
</dependency>
```

Import the following libraries:
```
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
```

Add `Logger.getRootLogger().setLevel(Level.WARN)` in the Main to make the output less verbose

Configure Spark app in Main:
```
val conf = new SparkConf().setAppName("My first Spark application")
val sc = new SparkContext(conf)
```

### Compile the app in a jar
```
mvn package
```
if it does not compile because of `-make:transitive`, just remove `<arg>-make:transitive</arg>` from the `pom.xml` file

if it does not compile because of `JUnitRunnner`, add the following dependency:
```
<dependency>
  <groupId>org.specs2</groupId>
  <artifactId>specs2-junit_${scala.compat.version}</artifactId>
  <version>2.4.16</version>
  <scope>test</scope>
</dependency>
```

### Test the scala app:
`java -cp {scala-library.jar}:{app jar} {app class}`

#### For example:
`java -cp /opt/spark/jars/scala-library-2.11.8.jar:target/bdproj-1.0-SNAPSHOT.jar lucnik.App`

### Deploy app on yarn
`/opt/spark2/bin/spark-submit --master yarn --class {app class} {app jar}`

### For example
``/opt/spark2/bin/spark-submit --master yarn --class lucnik.App target/bdproj-1.0-SNAPSHOT.jar`
