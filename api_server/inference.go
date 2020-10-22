package main

import (
	"bytes"
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

func init() {
	staticModel = &TransformerModel{}
	//LoadSavedModel时使用的go tensorflow版本不能低于tf.saved_model.builder.SavedModelBuilder时使用的tensorflow版本
	if model, err := tf.LoadSavedModel("../mymodel", []string{"classical2modern"}, nil); err == nil {
		staticModel.model = model
		staticModel.token2id, staticModel.id2token = loadVocab("../data/vocab_char.txt")

		inputTensor = staticModel.model.Graph.Operation("input_x").Output(0)
		decoderInput = staticModel.model.Graph.Operation("decoder_input").Output(0)
		isTraining = staticModel.model.Graph.Operation("is_training").Output(0)
		yPredict = staticModel.model.Graph.Operation("y_predict_v2").Output(0)

		//第一次执行model.Session.Run很耗时，所以初始化后先预热一下
		staticModel.Predict("翻译测试")
	} else {
		fmt.Printf("read transformer model file %s failed: %v", "classical2modern", err)
		return
	}
}

func GetModelInstance() *TransformerModel {
	return staticModel
}

func (model TransformerModel) Predict(text string) string {
	tokenIds := model.convertText2Id(text)
	input := [][]int32{tokenIds}

	decodeInput := []int32{model.token2id["<s>"]}
	_decodeInput := decodeInput
	for i := 0; i < 50; i++ {
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

	predictSentence := model.convertId2Token(predictIds)
	return predictSentence
}

func (model TransformerModel) runCalculation(X [][]int32, decodeInput [][]int32) [][]int32 {

	inputLayer := make(map[tf.Output]*tf.Tensor)

	inputLayer[inputTensor], _ = tf.NewTensor(X)

	inputLayer[decoderInput], _ = tf.NewTensor(decodeInput)

	inputLayer[isTraining], _ = tf.NewTensor(false)

	outputLayer := []tf.Output{
		//python版tensorflow/keras中定义的输出层output_layer
		yPredict,
	}
	result, err := model.model.Session.Run(inputLayer, outputLayer, nil)

	if err == nil {
		predict := result[0].Value().([][]int32)
		return predict
	} else {
		fmt.Printf("predict failed: %v", err)
		return nil
	}
}

func (model TransformerModel) convertText2Id(text string) []int32 {
	X := make([]int32, 0)
	for _, ch := range text {
		X = append(X, model.token2id[string(ch)])
	}
	X = append(X, model.token2id["<s>"])
	return X
}

func (model TransformerModel) convertId2Token(predictSentence []int32) string {
	var stringBuilder bytes.Buffer
	for _, tid := range predictSentence {
		if tid == model.token2id["</s>"] {
			break
		}
		stringBuilder.WriteString(model.id2token[tid])
	}
	return stringBuilder.String()
}
