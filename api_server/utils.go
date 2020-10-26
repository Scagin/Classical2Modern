package main

import (
	"bufio"
	"bytes"
	"io"
	"log"
	"os"
	"strings"
)

func loadVocab(filePath string) (map[string]int32, map[int32]string) {
	inputFile, inputError := os.Open(filePath)

	if inputError != nil {
		log.Println("打开文件时出错", inputError.Error())
		return nil, nil
	}
	defer inputFile.Close()

	i := int32(0)
	inputReader := bufio.NewReader(inputFile)
	token2id, id2token := make(map[string]int32), make(map[int32]string)
	for {
		inputString, readerError := inputReader.ReadString('\n')

		if readerError == io.EOF {
			break
		}

		token2id[strings.TrimSpace(inputString)] = i
		id2token[i] = strings.TrimSpace(inputString)
		i++
	}

	return token2id, id2token
}

func convertText2Id(text string, token2id map[string]int32) []int32 {
	X := make([]int32, 0)
	for _, ch := range text {
		X = append(X, token2id[string(ch)])
	}
	X = append(X, token2id["<s>"])
	return X
}

func convertId2Token(predictSentence []int32, token2id map[string]int32, id2token map[int32]string) string {
	var stringBuilder bytes.Buffer
	for _, tid := range predictSentence {
		if tid == token2id["</s>"] {
			break
		}
		stringBuilder.WriteString(id2token[tid])
	}
	return stringBuilder.String()
}
