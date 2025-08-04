import 'dart:ui' as ui;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:character_recognition_with_update/services/api_service.dart';

class CanvasScreen extends StatefulWidget {
  const CanvasScreen({super.key});

  @override
  State<CanvasScreen> createState() => _CanvasScreenState();
}

class _CanvasScreenState extends State<CanvasScreen> {
  List<Offset> points = [];
  GlobalKey canvasKey = GlobalKey();
  String prediction = '';
  double confidence = 0.0;

  void clearCanvas() {
    setState(() {
      points.clear();
      prediction = '';
      confidence = 0.0;
    });
  }

  Future<void> predictCharacter() async {
    try {
      RenderRepaintBoundary boundary =
      canvasKey.currentContext!.findRenderObject() as RenderRepaintBoundary;
      ui.Image canvasImage = await boundary.toImage(pixelRatio: 1.0);
      ByteData? byteData =
      await canvasImage.toByteData(format: ui.ImageByteFormat.png);
      Uint8List imageBytes = byteData!.buffer.asUint8List();

      var result = await ApiService.predictCharacter(imageBytes);

      setState(() {
        prediction = result['prediction'] ?? '';
        confidence = result['confidence'] ?? 0.0;
      });
    } catch (e) {
      print("Prediction error: $e");
    }
  }

  Future<void> correctCharacter() async {
    if (prediction.isEmpty) return;

    String? label = await showDialog<String>(
      context: context,
      builder: (_) {
        TextEditingController controller = TextEditingController();
        return AlertDialog(
          title: const Text('Enter Correct Label'),
          content: TextField(
            controller: controller,
            maxLength: 1,
            decoration: const InputDecoration(hintText: "a-z"),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, controller.text),
              child: const Text('Submit'),
            )
          ],
        );
      },
    );

    if (label == null || label.isEmpty) return;

    RenderRepaintBoundary boundary =
    canvasKey.currentContext!.findRenderObject() as RenderRepaintBoundary;
    ui.Image canvasImage = await boundary.toImage(pixelRatio: 1.0);
    ByteData? byteData =
    await canvasImage.toByteData(format: ui.ImageByteFormat.png);
    Uint8List imageBytes = byteData!.buffer.asUint8List();

    await ApiService.correctCharacter(imageBytes, label);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("Correction submitted")),
    );
  }

  @override
  Widget build(BuildContext context) {
    List<Widget> resultWidgets = [];

    if (prediction != "") {
      resultWidgets.addAll([
        const SizedBox(height: 10),
        Text("Prediction: $prediction"),
        Text("Confidence: ${confidence.toStringAsFixed(4)}"),
      ]);
    }

    return Scaffold(
      appBar: AppBar(title: const Text("Draw Character")),
      body: Column(
        children: [
          const SizedBox(height: 20),
          Center(
            child: RepaintBoundary(
              key: canvasKey,
              child: Container(
                width: 300,
                height: 300,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  color: Colors.white,
                ),
                child: GestureDetector(
                  onPanUpdate: (details) {
                    RenderBox box =
                    canvasKey.currentContext!.findRenderObject() as RenderBox;
                    Offset point = box.globalToLocal(details.globalPosition);
                    setState(() {
                      points.add(point);
                    });
                  },
                  onPanEnd: (_) => points.add(Offset.infinite),
                  child: CustomPaint(painter: DrawingPainter(points)),
                ),
              ),
            ),
          ),
          const SizedBox(height: 20),
          ElevatedButton(
            onPressed: predictCharacter,
            child: const Text("Predict"),
          ),
          ...resultWidgets,
          const SizedBox(height: 10),
          ElevatedButton(
            onPressed: correctCharacter,
            child: const Text("Submit Correction"),
          ),
          const SizedBox(height: 10),
          ElevatedButton(
            onPressed: clearCanvas,
            child: const Text("Clear"),
          ),
        ],
      ),
    );
  }

}

class DrawingPainter extends CustomPainter {
  final List<Offset> points;

  DrawingPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    Paint paint = Paint()
      ..color = Colors.black // black pen
      ..strokeWidth = 10
      ..strokeCap = StrokeCap.round;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != Offset.infinite && points[i + 1] != Offset.infinite) {
        canvas.drawLine(points[i], points[i + 1], paint);
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
