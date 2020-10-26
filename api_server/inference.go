package main

import (
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type TransformerModel struct {
	model    *tf.SavedModel
	token2id map[string]int32
	id2token map[int32]string
}

var (
	staticModel  *TransformerModel
	inputTensor  tf.Output
	decoderInput tf.Output
	isTraining   tf.Output
	yPredict     tf.Output
)

func initModel() {
	// Load model
	staticModel = &TransformerModel{}
	model, err := tf.LoadSavedModel(modelPath, []string{"classical2modern"}, nil)

	if err != nil {
		fmt.Printf("Read transformer model file %s failed: %v", "classical2modern", err)
		return
	}

	staticModel.model = model

	// Load vocabulary
	staticModel.token2id, staticModel.id2token = loadVocab(vocabPath)

	// Get Tensor Operation
	inputTensor = staticModel.model.Graph.Operation("input_x").Output(0)
	decoderInput = staticModel.model.Graph.Operation("decoder_input").Output(0)
	isTraining = staticModel.model.Graph.Operation("is_training").Output(0)
	yPredict = staticModel.model.Graph.Operation("y_predict_v2").Output(0)

	//第一次执行model.Session.Run很耗时，所以初始化后先预热一下
	staticModel.predict("床前明月光")
}

func getModelInstance() *TransformerModel {
	return staticModel
}

func (model TransformerModel) predict(text string) string {
	tokenIds := convertText2Id(text, model.token2id)
	input := [][]int32{tokenIds}

	decodeInput := []int32{model.token2id["<s>"]}
	_decodeInput := decodeInput
	for i := 0; i < maxLength; i++ {
		_predict := model.runCalculation(input, [][]int32{_decodeInput})
		lastToken := _predict[0][len(_predict[0])-1]
		if lastToken == model.token2id["<pad>"] || lastToken == model.token2id["</s>"] {
			break
		}
		_decodeInput = append(decodeInput, _predict[0]...)
	}
	predictIds := _decodeInput[1:]

	//_predict := model.runCalculation(input)
	//
	//predictSentence := model.convertId2Token(_predict[0])

	predictSentence := convertId2Token(predictIds, model.token2id, model.id2token)
	return predictSentence
}

func (model TransformerModel) runCalculation(X [][]int32, decodeInput [][]int32) [][]int32 {
	inputLayer := make(map[tf.Output]*tf.Tensor)

	inputLayer[inputTensor], _ = tf.NewTensor(X)
	inputLayer[decoderInput], _ = tf.NewTensor(decodeInput)
	inputLayer[isTraining], _ = tf.NewTensor(false)

	outputLayer := []tf.Output{yPredict}

	result, err := model.model.Session.Run(inputLayer, outputLayer, nil)

	if err != nil {
		fmt.Printf("predict failed: %v", err)
		return nil
	}

	predict := result[0].Value().([][]int32)
	return predict
}
