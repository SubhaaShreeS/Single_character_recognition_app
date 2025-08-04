import 'dart:convert';
import 'dart:typed_data';
import 'package:dio/dio.dart';

class ApiService {
  static const String baseUrl = "http://test0.gpstrack.in:9010";

  static Future<Map<String, dynamic>> predictCharacter(Uint8List imageBytes) async {
    try {
      FormData formData = FormData.fromMap({
        "image": MultipartFile.fromBytes(imageBytes, filename: "char.png"),
      });

      Response response = await Dio().post("$baseUrl/predict", data: formData);
      return response.data;
    } catch (e) {
      return {"error": e.toString()};
    }
  }

  static Future<void> correctCharacter(Uint8List imageBytes, String label) async {
    try {
      FormData formData = FormData.fromMap({
        "image": MultipartFile.fromBytes(imageBytes, filename: "char.png"),
        "label": label,
      });

      await Dio().post("$baseUrl/correct", data: formData);
    } catch (e) {
      // Handle error if needed
    }
  }
}
