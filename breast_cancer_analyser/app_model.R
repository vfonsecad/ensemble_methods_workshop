#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

# Libraries
required_packages<-c("data.table", "shiny", "ggplot2")
packages_test<-(required_packages%in%rownames(installed.packages())==FALSE)
if(any(packages_test)) install.packages(required_packages[which(packages_test)])


library(data.table)
library(shiny)
library(ggplot2)



# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel(" - Breast cancer analyser - "),
  
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
                 img(src='logo_heart.png', width=350, height=250)
    ),
    
    # Result in main panel
    mainPanel(
      tabsetPanel(
        tabPanel(title="MODEL SUMMARY",
                 actionButton("check_model_summary","Check"),
                 verbatimTextOutput("trained_model_verbatim"),
                 tableOutput("summary_performance_table"),
                 plotOutput("output_varimp_plot")
                 
        ),

        tabPanel("PREDICTIONS SIMULATOR",
                 br(),
                 h4("Input data format example"),
                 verbatimTextOutput("new_data_example"),
                 fileInput("new_data_filename", "Select new input data file .csv", accept = c(".csv")),
                 br(),
                 actionButton("get_new_data",
                              "Get predictions"),
                 textOutput("uploaded_newdata_ready"),
                 plotOutput("output_pred_plot"),
                 br(),
                 br(),
                 br(),
                 downloadButton("predict_predset_download", label = "Download predictions")
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
  
  
  ########################################################################   
  ## TAB PANEL 1. Dashboard tab
  ########################################################################   
  
 
  dashboard_summary_model<-eventReactive(input$check_model_summary,{
    
    output_verbatim <- uploaded_model_list()$model_output[["trained_model"]]
    output_summary_table <- uploaded_model_list()$model_output[["summary_performance"]]
    output_varimp <- uploaded_model_list()$model_output[["variable_importance"]]
    
    
    output_varimp_plot <- ggplot(output_varimp, aes(y = Variable, x = Tree)) + 
                              geom_point(aes(colour = Importance), size = 5)
                              theme(axis.text.x = element_text(size = 12, colour = "black", family="Times"),
                                   axis.text.y = element_text(size = 12, colour = "black", family="Times"))
    
                            
    output <- list(output_verbatim = output_verbatim,
                   output_summary_table = output_summary_table,
                   output_varimp_plot = output_varimp_plot)
    return(output)
  })
  
  output$trained_model_verbatim <- renderPrint({dashboard_summary_model()[["output_verbatim"]]})
  output$summary_performance_table <- renderTable({dashboard_summary_model()[["output_summary_table"]]})
  output$output_varimp_plot <- renderPlot({dashboard_summary_model()[["output_varimp_plot"]]})

  ########################################################################   
  ## TAB PANEL 2.  Predict new series based on input data
  ########################################################################   
  
  input_data_example <- reactive({

    vars_input <- colnames(uploaded_model_list()$model_output$trained_model$trainingData)[-1]
    row_input <- uploaded_model_list()$model_output$trained_model$trainingData[1,-1]

    return(list(vars_names = paste0(c("id",vars_input), collapse = ";"),
               row_example = paste0(c("0rt51",row_input), collapse = ";")))


  })

  output$new_data_example <- renderPrint({cat(input_data_example()[["vars_names"]]);
                                          cat("\n");
                                          cat(input_data_example()[["row_example"]])})

  
  get_predictions <- eventReactive(input$get_new_data,{
    
    current_test_data <- fread(input$new_data_filename$datapath, sep = ";")
    current_predictions <-predict(uploaded_model_list()$model_output[["trained_model"]],current_test_data, type = "raw")
    
    current_test_data[["Predictions"]] <- as.factor(current_predictions)
    
    pred_plot <- ggplot(current_test_data, aes(x = Predictions))+
      geom_bar(colour = "purple", fill = "purple")
    
    output <- list(pred_data = current_test_data, pred_plot = pred_plot, ok = "Finished")
    return(output)
  })
  
  output$uploaded_newdata_ready<-renderText({get_predictions()$ok})
  
  output$output_pred_plot <- renderPlot({get_predictions()$pred_plot})
  
  output$predict_predset_download <- downloadHandler(
    filename = function() {
      paste0('new_data_predictions','.csv')
    },
    content = function(con) {
      write.csv(get_predictions()$pred_data, con)
    }
  )
  }

# Run the application 
shinyApp(ui = ui, server = server)