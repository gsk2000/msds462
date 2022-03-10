//
//  ViewController.swift
//  numberRecog
//
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var detectedText: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var correctedImageView: UIImageView!
    
    var model: VNCoreMLModel!
    
    var textMetadata = [Int: [Int: String]]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        loadModel()
        activityIndicator.hidesWhenStopped = true
    }
    
    private func loadModel() {
        model = try? VNCoreMLModel(for: model4().model)
    }

    // MARK: IBAction
    
    @IBAction func pickImageClicked(_ sender: UIButton) {
        let alertController = createActionSheet()
        let action1 = UIAlertAction(title: "Camera", style: .default, handler: {
            (alert: UIAlertAction!) -> Void in
            self.showImagePicker(withType: .camera)
        })
        let action2 = UIAlertAction(title: "Photos", style: .default, handler: {
            (alert: UIAlertAction!) -> Void in
            self.showImagePicker(withType: .photoLibrary)
        })
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        addActionsToAlertController(controller: alertController,
                                    actions: [action1, action2, cancelAction])
        self.present(alertController, animated: true, completion: nil)
    }
    
    // MARK: image picker
    
    func showImagePicker(withType type: UIImagePickerControllerSourceType) {
        let pickerController = UIImagePickerController()
        pickerController.delegate = self
        pickerController.sourceType = type
        present(pickerController, animated: true)
    }
    
    func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [String : Any]) {
        dismiss(animated: true)
        guard let image = info[UIImagePickerControllerOriginalImage] as? UIImage else {
            fatalError("Couldn't load image")
        }
        let newImage = fixOrientation(image: image)
        self.imageView.image = newImage
        clearOldData()
        showActivityIndicator()
        DispatchQueue.global(qos: .userInteractive).async {
            self.detectText(image: newImage)
        }
    }
    
    // MARK: text detection
    
    func detectText(image: UIImage) {
         //let convertedImage = image |> adjustColors |> convertToGrayscale
        let convImage = image |> invertColors |> adjustColors
        let convertedImage = image
        // Show the pre-processed image
        //DispatchQueue.main.async {
            //self.correctedImageView.image = convertedImage
        //}
        let handler = VNImageRequestHandler(cgImage: convertedImage.cgImage!)
        let request: VNDetectTextRectanglesRequest =
            VNDetectTextRectanglesRequest(completionHandler: { [unowned self] (request, error) in
            if (error != nil) {
                print("Got Error In Run Text Dectect Request :(")
            } else {
                guard let results = request.results as? Array<VNTextObservation> else {
                    fatalError("Unexpected result type from VNDetectTextRectanglesRequest")
                }
                if (results.count == 0) {
                    self.handleEmptyResults()
                    return
                }
                // Verify detected rectangle is valid.
                //let boundingBox = results.boundingBox.scaled(to: imageSize)
                //guard image.extent.contains(boundingBox)
                //    else { print("invalid detected rectangle"); return }
               //----------------------
                UIGraphicsBeginImageContextWithOptions(image.size, true, 0)
                       
                        let context = UIGraphicsGetCurrentContext()
                        context?.setStrokeColor(UIColor.red.cgColor)
                        context?.translateBy(x: 0, y: image.size.height)
                        context?.scaleBy(x: 1, y: -1)
                        context?.draw(image.cgImage!, in: CGRect(origin: .zero, size: image.size))
                       
                        for result in results {
                            if let textObservation = result as? VNTextObservation {
                                let rect: CGRect = {
                                    var rect = CGRect()
                                    rect.origin.x = textObservation.boundingBox.origin.x * image.size.width
                                    rect.origin.y = textObservation.boundingBox.origin.y * image.size.height
                                    rect.size.width = textObservation.boundingBox.size.width * image.size.width
                                    rect.size.height = textObservation.boundingBox.size.height * image.size.height
                                    return rect
                                }()
                               
                                context?.stroke(rect, width: 5)
                               
                                //print(textObservation)
                            }
                        }
                       
                        let drawnImage = UIGraphicsGetImageFromCurrentImageContext()
                        UIGraphicsEndImageContext()
                DispatchQueue.main.async {
                        self.imageView.image = drawnImage
                }
                   // -------------
                var numberOfWords = 0
                for textObservation in results {
                    var numberOfCharacters = 0
                    for rectangleObservation in textObservation.characterBoxes! {
                        let croppedImage = crop(image: convImage, rectangle: rectangleObservation)
                        if let croppedImage = croppedImage {
                            let processedImage = preProcess(image: croppedImage)
                            DispatchQueue.main.async {
                                self.correctedImageView.image = processedImage
                            }
                            self.classifyImage(image: processedImage,
                                               wordNumber: numberOfWords,
                                               characterNumber: numberOfCharacters)
                            numberOfCharacters += 1
                        }
                    }
                    numberOfWords += 1
                }
            }
        })
        request.reportCharacterBoxes = true
        do {
            try handler.perform([request])
        } catch {
            print(error)
        }
    }
    
    func handleEmptyResults() {
        DispatchQueue.main.async {
            self.hideActivityIndicator()
            self.detectedText.text = "The image does not contain any text."
        }
        
    }
    
    func classifyImage(image: UIImage, wordNumber: Int, characterNumber: Int) {
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                let topResult = results.first else {
                    fatalError("Unexpected result type from VNCoreMLRequest")
            }
            let result = topResult.identifier
            let classificationInfo: [String: Any] = ["wordNumber" : wordNumber,
                                                     "characterNumber" : characterNumber,
                                                     "class" : result]
            self?.handleResult(classificationInfo)
        }
        guard let ciImage = CIImage(image: image) else {
            fatalError("Could not convert UIImage to CIImage :(")
        }
        let handler = VNImageRequestHandler(ciImage: ciImage)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            }
            catch {
                print(error)
            }
        }
    }
    
    func handleResult(_ result: [String: Any]) {
        objc_sync_enter(self)
        guard let wordNumber = result["wordNumber"] as? Int else {
            return
        }
        guard let characterNumber = result["characterNumber"] as? Int else {
            return
        }
        guard let characterClass = result["class"] as? String else {
            return
        }
        if (textMetadata[wordNumber] == nil) {
            let tmp: [Int: String] = [characterNumber: characterClass]
            textMetadata[wordNumber] = tmp
        } else {
            var tmp = textMetadata[wordNumber]!
            tmp[characterNumber] = characterClass
            textMetadata[wordNumber] = tmp
        }
        objc_sync_exit(self)
        DispatchQueue.main.async {
            self.hideActivityIndicator()
            self.showDetectedText()
        }
    }
    
    func showDetectedText() {
        var result: String = ""
        if (textMetadata.isEmpty) {
            detectedText.text = "The image does not contain any text."
            return
        }
        let sortedKeys = textMetadata.keys.sorted()
        for sortedKey in sortedKeys {
            result +=  word(fromDictionary: textMetadata[sortedKey]!) + " "
        }
        detectedText.text = result
    }
    
    func word(fromDictionary dictionary: [Int : String]) -> String {
        let sortedKeys = dictionary.keys.sorted()
        var word: String = ""
        for sortedKey in sortedKeys {
            let char: String = dictionary[sortedKey]!
            word += char
        }
        return word
    }
    
    // MARK: private
    
    private func clearOldData() {
        detectedText.text = ""
        textMetadata = [:]
    }
    
    private func showActivityIndicator() {
        activityIndicator.startAnimating()
    }
    
    private func hideActivityIndicator() {
        activityIndicator.stopAnimating()
    }


}

