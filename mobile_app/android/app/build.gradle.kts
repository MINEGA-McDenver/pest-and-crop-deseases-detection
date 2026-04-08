plugins {
    id("com.android.application")
    id("kotlin-android")
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.pestdetection.crop_doctor"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_11.toString()
    }

    defaultConfig {
        applicationId = "com.pestdetection.crop_doctor"
        minSdk = 24
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
            val keystoreFile = System.getenv("CD_UPLOAD_STORE_FILE")
            if (!keystoreFile.isNullOrBlank()) {
                signingConfig = signingConfigs.create("release") {
                    storeFile = file(keystoreFile)
                    storePassword = System.getenv("CD_UPLOAD_STORE_PASSWORD")
                    keyAlias = System.getenv("CD_UPLOAD_KEY_ALIAS")
                    keyPassword = System.getenv("CD_UPLOAD_KEY_PASSWORD")
                }
            }
            // Disable R8 for pilot builds to avoid optional TensorFlow GPU class stripping issues.
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }

    aaptOptions {
        noCompress += "tflite"
    }
}

flutter {
    source = "../.."
}

dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
}