Legal Assistant Chatbot - Flutter Frontend
This is the official Flutter-based frontend for the Madhya Pradesh Legal Assistant Chatbot. It provides a clean, responsive user interface for interacting with the powerful RAG backend, allowing users to ask questions and receive source-cited answers in real-time.

This guide provides all the necessary steps to set up the Flutter environment, configure the application, and run it on a web browser or an Android device.

📸 Application Preview
📋 Prerequisites
Before you begin, ensure you have the following:

Backend Server Running: The Python backend must be running and accessible on your local network. Please follow the instructions in the main project README.md to start the server.

Flutter SDK: You need the Flutter SDK installed on your system.

Code Editor: A code editor with Flutter support, such as VS Code with the Flutter extension or Android Studio.

(For Android APK): The Java Development Kit (JDK) and the full Android Studio installation are required to build and install the Android app.

🚀 Setup Instructions
Follow these steps to get the Flutter application configured and ready to run.

Step 1: Get the Project Code
Download or clone the project repository to your local machine.

Step 2: Install Flutter
If you don't have Flutter installed, follow the official documentation for your operating system. This is a one-time setup.

Install Flutter on Windows

Install Flutter on macOS

Install Flutter on Linux

After installation, run the following command in your terminal to verify that everything is set up correctly. Address any issues reported by the doctor.

flutter doctor

Step 3: Install App Dependencies
Navigate to the project directory in your terminal and run the following command to fetch all the necessary packages for the app.

flutter pub get

Step 4: 🔌 Configure the Backend URL (CRITICAL)
This is the most important step. You need to tell the Flutter app where to find your running backend server.

Open the project in your code editor.

Navigate to the file: lib/services/chat_service.dart.

Find the line that defines the _baseUrl.

Update the IP address to match the local network IP of the computer running the Python backend.

// lib/services/chat_service.dart

import 'package:http/http.dart' as http;
import 'dart:convert';

class ChatService {
  // 🔽🔽🔽 CHANGE THIS LINE 🔽🔽🔽
  final String _baseUrl = "[http://192.168.29.152:8000](http://192.168.29.152:8000)"; // Use your computer's local IP
  // 🔼🔼🔼 CHANGE THIS LINE 🔼🔼🔼

  Future<Map<String, dynamic>> sendMessage(String query) async {
    // ... rest of the code
  }
}

How to find your computer's local IP?

Windows: Open Command Prompt (cmd) and type ipconfig. Find the "IPv4 Address".

macOS/Linux: Open a terminal and type ifconfig or ip a. Find the "inet" address.

🏃 Running the Application
Once setup is complete, you can run the app in two ways.

Option A: 🖥️ Run in a Web Browser
This is the quickest way to test the application. Run the following command in your terminal from the project root:

flutter run -d chrome

A new Chrome window will open with the chatbot application running. You can make changes to the code, and the app will "hot reload" instantly.

Option B: 📱 Build and Run on an Android Phone
To run the app on a physical Android device, follow these steps:

Enable Developer Options: On your Android phone, go to Settings > About phone and tap on Build number seven times. This will enable "Developer options".

Enable USB Debugging: Go to Settings > System > Developer options and turn on "USB debugging".

Connect Your Phone: Connect your phone to your computer with a USB cable. Authorize the connection on your phone if prompted.

Run the App: In your terminal, simply run:

flutter run

Flutter will automatically detect your connected device, build the app, and install it.

Build an APK File (for sharing): To create a standalone .apk file that you can share and install on any Android phone, run:

flutter build apk --release
