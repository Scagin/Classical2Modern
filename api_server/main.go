package main

import (
	"flag"
	"fmt"
	"github.com/gin-gonic/gin"
	"net/http"
)

func translation(text string) string {
	model := getModelInstance()
	outputSentence := model.predict(text)
	return outputSentence
}

var (
	modelPath string
	vocabPath string
	maxLength int
	port      int
)

func init() {
	flag.StringVar(&modelPath, "model_path", "../mymodel", "Path of you model directory.")
	flag.StringVar(&vocabPath, "vocab_path", "../data/vocab_char.txt", "Path of you vocabulary file.")
	flag.IntVar(&maxLength, "max_length", 120, "Max length of target sentences.")
	flag.IntVar(&port, "port", 9391, "Listen port of api server.")
}

func main() {
	flag.Parse()

	initModel()

	r := gin.Default()

	r.POST("/translation", func(c *gin.Context) {
		text := c.PostForm("text")
		c.JSON(http.StatusOK, gin.H{
			"result": translation(text),
		})
	})

	r.Run(fmt.Sprintf("0.0.0.0:%d", port))

}
