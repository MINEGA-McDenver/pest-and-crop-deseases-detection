import 'package:flutter_test/flutter_test.dart';
import 'package:crop_doctor/main.dart';
import 'package:flutter/material.dart';

void main() {
  testWidgets('App starts', (WidgetTester tester) async {
    await tester.pumpWidget(const CropDoctorApp());
    await tester.pump(const Duration(milliseconds: 300));
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
