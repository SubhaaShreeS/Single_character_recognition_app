import 'package:flutter/material.dart';
import '../screens/home_screen.dart';

void main() {
  runApp(const AlphabetApp());
}

class AlphabetApp extends StatelessWidget {
  const AlphabetApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Alphabet Recognition',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
        fontFamily: 'RobotoBlack',
      ),
      debugShowCheckedModeBanner: false,
      home: const HomeScreen(),
    );
  }
}
