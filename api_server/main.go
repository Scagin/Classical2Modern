package main

import (
	"github.com/gin-gonic/gin"
	"net/http"
)

func translation(text string) string {
	model := GetModelInstance()
	outputSentence := model.Predict(text)
	return outputSentence
}

func main() {

	r := gin.Default()

	r.POST("/translation", func(c *gin.Context) {
		text := c.PostForm("text")
		c.JSON(http.StatusOK, gin.H{
			"result": translation(text),
		})
	})

	r.Run("0.0.0.0:19267")

}
