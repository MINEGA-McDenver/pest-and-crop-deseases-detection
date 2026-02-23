import 'package:flutter_test/flutter_test.dart';
import 'package:crop_doctor/main.dart';

void main() {
  testWidgets('App starts', (WidgetTester tester) async {
    await tester.pumpWidget(const CropDoctorApp());
    expect(find.text('Crop Doctor'), findsOneWidget);
  });
}
