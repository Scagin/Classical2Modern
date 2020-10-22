package main

import (
	"bufio"
	"io"
	"log"
	"os"
	"strings"
)

func loadVocab(filePath string) (map[string]int32, map[int32]string) {
	inputFile, inputError := os.Open(filePath)
	if inputError != nil {
		log.Println("打开文件时出错", inputError.Error())
		return nil, nil // 退出函数
	}
	defer inputFile.Close()
	inputReader := bufio.NewReader(inputFile)
	i := int32(0)
	token2id := make(map[string]int32)
	id2token := make(map[int32]string)
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
