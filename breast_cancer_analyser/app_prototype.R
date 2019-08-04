#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

# Libraries
required_packages<-c("data.table", "shiny", "plotly")
packages_test<-(required_packages%in%rownames(installed.packages())==FALSE)
if(any(packages_test)) install.packages(required_packages[which(packages_test)])

library(kernlab)
library(data.table)
library(shiny)
library(plotly)



# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("PROTOTYPE OF A BASIC MACHINE THAT LEARNED"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(width=4,h4("UPLOAD MODEL"),
                 fileInput("model_filename", "Select model file .RData", accept = c(".RData")),
                 br(),
                 actionButton("check_upload_model_list",
                              "Check model"),
                 textOutput("uploaded_model_list_ready"),
                 br(),
                 br(),
                 img(src='logo.png', width=350, height=250)
    ),
    
    # Result in main panel
    mainPanel(
      tabsetPanel(
        tabPanel(title="MODEL SUMMARY",
                 uiOutput("model_list_input"),
                 actionButton("check_chosen_model","Check model")
                 
        ),

        tabPanel("PREDICTIONS SIMULATOR",
                 br(),
                 numericInput("var1", "X1", "0",width = '20%'),
                 numericInput("var1", "X1", "0",width = '20%'),
                 numericInput("var1", "X1", "0",width = '20%'),
                 actionButton("calculate_new_predictions","Go!"),
                 br(),
                 br(),
                 br(),
                 downloadButton("predict_predset2", label = "Download predictions")
        )
      )
    )
  )
)


server <- function(input, output) {
  
  
  ########################################################################   
  ## SIDEBAR:  Receive model
  ########################################################################   
  
  uploaded_model_list<-eventReactive(input$check_upload_model_list,{
    load(input$model_filename$datapath)
    ok <- ifelse(exists("model_output"),"Model ready", "The name of the model list must be model_output")
    output <- list(model_output = model_output, ok = ok)
    return(output)
  })
  output$uploaded_model_list_ready<-renderText({uploaded_model_list()$ok})
  
  
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)