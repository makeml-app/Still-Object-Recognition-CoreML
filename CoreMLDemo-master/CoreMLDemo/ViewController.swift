import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UINavigationControllerDelegate {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var classifier: UILabel!
    
    //var model_2: ObjectDetector!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }
    
    override func viewWillAppear(_ animated: Bool) {
        //model_2 = ObjectDetector()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func camera(_ sender: Any) {
        
        if !UIImagePickerController.isSourceTypeAvailable(.camera) {
            return
        }
        
        let cameraPicker = UIImagePickerController()
        cameraPicker.delegate = self
        cameraPicker.sourceType = .camera
        cameraPicker.allowsEditing = false
        
        present(cameraPicker, animated: true)
    }
    
    @IBAction func openLibrary(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.allowsEditing = false
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }

}

extension ViewController: UIImagePickerControllerDelegate {
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        
        picker.dismiss(animated: true)
        classifier.text = ""
        guard let image = info["UIImagePickerControllerOriginalImage"] as? UIImage else {
            return
        } //1
        
        imageView.contentMode = .scaleAspectFit
        imageView.image = image
        
        checkDifferentPredictions(image: image)
    }
    
    func checkDifferentPredictions(image: UIImage) {
        
        for view in imageView.subviews {
            view.removeFromSuperview()
        }
        
        if let sublayers = imageView.layer.sublayers {
            for sublayer in sublayers {
                sublayer.removeFromSuperlayer()
            }
        }
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 416, height: 416), true, 2.0)
        image.draw(in: CGRect(x: 0, y: 0, width: 416, height: 416))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(newImage.size.width), Int(newImage.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else {
            return
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(newImage.size.width), height: Int(newImage.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) //3
        
        context?.translateBy(x: 0, y: newImage.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        newImage.draw(in: CGRect(x: 0, y: 0, width: newImage.size.width, height: newImage.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer!, orientation: CGImagePropertyOrientation.downMirrored, options: [:])
        
        guard let modelURL = Bundle.main.url(forResource: "ObjectDetector", withExtension: "mlmodelc") else {
            return
        }
        
        let visionModel = try! VNCoreMLModel(for: MLModel(contentsOf: modelURL))
        let objectRecognition = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
            DispatchQueue.main.async(execute: {
                // perform all the UI updates on the main queue
                if let results = request.results {
                    
                    for observation in results where observation is VNRecognizedObjectObservation {
                        guard let objectObservation = observation as? VNRecognizedObjectObservation else {
                            continue
                        }
                        // Select only the label with the highest confidence.
                        
                        var width: CGFloat = 0.0
                        var height: CGFloat = 0.0
                        var xIncrement: CGFloat = 0.0
                        var yIncrement: CGFloat = 0.0
                        
                        if image.size.width > image.size.height {
                            width = self.imageView.bounds.size.width
                            height = self.imageView.bounds.size.width * (image.size.height) / (image.size.width)
                            yIncrement = (self.imageView.bounds.size.height - height) / 2.0
                        } else {
                            height = self.imageView.bounds.size.height
                            width = self.imageView.bounds.size.height * (image.size.width) / (image.size.height)
                            xIncrement = (self.imageView.bounds.size.width - width) / 2.0
                        }
                        
                        let topLabelObservation = objectObservation.labels[0]
                        var objectBounds = VNImageRectForNormalizedRect(objectObservation.boundingBox, Int(width), Int(height))
                        
                        objectBounds.origin.x = objectBounds.origin.x + xIncrement
                        objectBounds.origin.y = objectBounds.origin.y + yIncrement
                        
                        let shapeLayer = self.createRoundedRectLayerWithBounds(objectBounds)
                        
                        let textLayer = self.createTextSubLayerInBounds(objectBounds,
                                                                        identifier: topLabelObservation.identifier,
                                                                        confidence: topLabelObservation.confidence)
                        
                        shapeLayer.addSublayer(textLayer)
                        self.imageView.layer.addSublayer(shapeLayer)
                    }
                }
            })
        })
        
        try! imageRequestHandler.perform([objectRecognition])
    }
    
    func createTextSubLayerInBounds(_ bounds: CGRect, identifier: String, confidence: VNConfidence) -> CATextLayer {
        let textLayer = CATextLayer()
        textLayer.name = "Object Label"
        let formattedString = NSMutableAttributedString(string: String(format: "\(identifier)\nConfidence:  %.2f", confidence))
        let largeFont = UIFont(name: "Helvetica", size: 24.0)!
        formattedString.addAttributes([NSAttributedString.Key.font: largeFont], range: NSRange(location: 0, length: identifier.count))
        formattedString.addAttributes([NSAttributedString.Key.foregroundColor: UIColor.white], range: NSRange(location: 0, length: formattedString.length))
        textLayer.string = formattedString
        textLayer.bounds = CGRect(x: 0, y: 0, width: bounds.size.width - 10, height: bounds.size.height - 10)
        textLayer.position = CGPoint(x: bounds.midX + 10, y: bounds.midY)
        textLayer.shadowOpacity = 0.7
        textLayer.shadowOffset = CGSize(width: 2, height: 2)
        textLayer.foregroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [1.0, 1.0, 1.0, 1.0])
        textLayer.contentsScale = 2.0 // retina rendering
        // rotate the layer into screen orientation and scale and mirror
        //textLayer.setAffineTransform(CGAffineTransform.sca.scaledBy(x: 1.0, y: -1.0))
        return textLayer
    }
    
    func createRoundedRectLayerWithBounds(_ bounds: CGRect) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        shapeLayer.name = "Found Object"
        shapeLayer.backgroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.2, 0.2, 1.0, 0.4])
        shapeLayer.cornerRadius = 7
        return shapeLayer
    }
}
