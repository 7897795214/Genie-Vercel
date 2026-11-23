import AppKit
import PDFKit
import Vision
import Foundation

guard CommandLine.arguments.count > 1,
      let doc = PDFDocument(url: URL(fileURLWithPath: CommandLine.arguments[1])) else {
    fputs("Could not open PDF for OCR.\n", stderr)
    exit(2)
}
if doc.pageCount > 20 {
    fputs("OCR is limited to the first 20 pages.\n", stderr)
}
for index in 0..<min(doc.pageCount, 20) {
    guard let page = doc.page(at: index) else { continue }
    let bounds = page.bounds(for: .mediaBox)
    let scale = min(2.0, 2200.0 / max(bounds.width, bounds.height))
    let size = CGSize(width: bounds.width * scale, height: bounds.height * scale)
    let image = page.thumbnail(of: size, for: .mediaBox)
    var rect = CGRect(origin: .zero, size: image.size)
    guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else { continue }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    do {
        try VNImageRequestHandler(cgImage: cgImage).perform([request])
        for item in request.results ?? [] {
            if let line = item.topCandidates(1).first?.string { print(line) }
        }
    } catch {
        fputs("OCR failed on page \(index + 1): \(error.localizedDescription)\n", stderr)
    }
}
