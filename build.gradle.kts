plugins {
    kotlin("jvm") version "2.1.10"
}

group = "org.creamycookie"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    val jeneticsVersion = "8.1.0"

    implementation("io.jenetics:jenetics:$jeneticsVersion")
    implementation("io.jenetics:jenetics.prog:$jeneticsVersion")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(21)
}