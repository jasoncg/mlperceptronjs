/*
 * mlperceptron.js
 * 2015-02-11
 *
 * jasoncg
 *
 */
/*
 * Takes an array of weights, and a reference to the layer
 */
function Neuron(weights, perceptron_layer) {
  var self=this;
  
  this.weights = weights;
  this.perceptron = perceptron_layer;
  this.dropout = false;
  //Variable for training
  this.last_guessed=null;
  this.error=null;
  //console.log('Generate Neuron', perceptron_layer.index, weights);
  
  this.sigmoid = function(input) {
    //return 1/(1+Math.exp(-input));
    return Math.tanh(input);
  }
  this.disigmoid = function(input) {
    //var s=self.sigmoid(input);
    //return s*(1-s);
    return 1-Math.pow(Math.tanh(input),2);
  }
  //Performs the normal evaluation, but returns boolean instead
  this.evaluate_output=function(inputs) {
    this.evaluate(inputs);
    
    if(self.last_guessed<0)
      return 0;
    return 1;
  }
  /**
   * Evaluate the array of inputs
   */
  this.evaluate=function(inputs) {
    var result=0;
    self.dropout = false;
    
    //Process the bias
    result+=self.weights[0];
    
    var i=1;
    for(i=1;i<weights.length;i++) {
      result+=inputs[i-1]*self.weights[i];
    }
    self.last_guessed=self.sigmoid(result);
    //console.log('Neuron eval', result);
    return self.last_guessed;
  }
  
  this.calculate_delta=function(expected) {
    if(expected<=0)
      expected=-1;
    var err=expected-self.last_guessed;
    //Sum of squares
    //err=err*err*0.5;
    
    return err;
  }
  
  this.backpropagate=function(err) {
    this.error=err*self.disigmoid(self.last_guessed);
    
    return self.error;
  }
  
  this.backpropagate_update_weights=function(inputs, learning_rate) {
    //First, update the bias
    self.weights[0]=self.weights[0]+learning_rate*self.error;
    
    for(var i=1;i<self.weights.length;i++) {
      self.weights[i]=self.weights[i]+learning_rate*self.error*inputs[i-1];
    }
    
    return self.evaluate(inputs);
  }
}

function Perceptron(input_count, neuron_count, weights, layer_prev) {
  var self=this;
  
  this.index=0;
  this.neurons=[];//new Array(neuron_count);
  this.layer_prev=layer_prev;
  this.layer_next=null;
  
  if(typeof(this.layer_prev)=='undefined')
    this.layer_prev=null;
  
  if(layer_prev!=null)
    this.index=layer_prev.index+1;
  
  //Build neurons
  var weight_index=0;
  var weights_per_neuron=input_count+1; //Add one for bias
  for(var i=0;i<neuron_count;i++) {
   // console.log('Create Neuron ',i);
    var n=new Neuron(weights.subarray(weight_index, weight_index+weights_per_neuron),
                    this);
    this.neurons.push(n);
    weight_index+=weights_per_neuron;
  }
  
  /**
   * Add a layer to the end of the network
   * neuron_count is indicative of the number of neurons, and consequently indicates
   * the number of outputs from this layer.
   * If weights is not provided then random weights will be generated.
   */
  this.add_next_layer=function(neuron_count, weights) {
    if(self.layer_next!=null)
      return self.layer_next.add_next_layer(neuron_count, weights);
    
    if(weights==null||typeof(weights)=='undefined') {
      //Generate random weights if no weights provided
      weights=Perceptron.GenerateWeights(self.neurons.length, neuron_count);
    }
    self.layer_next=new Perceptron(self.neurons.length, neuron_count, weights, this);
    
    return self.layer_next;
  }
  
  this.evaluateb=function(inputs) {
    var results=self.evaluate(inputs);
    
    var output=new Int8Array(results.length);
    for(var i=0;i<results.length;i++) {
      if(results[i]<0.5)
       output[i]=0;
      else
       output[i]=1;
    }
    
    return output;
  }
  this.evaluate=function(inputs, dropout_rate) {
    //console.log('Eval layer', self.index, '[',self.neurons.length,']', inputs);
    //var buffer = new ArrayBuffer(self.neurons.length*32);
    var results= new Float32Array(self.neurons.length);//buffer);
    //console.log('Result Buffer Size', results.length);//, buffer.byteLength);
    //Evaulate this layer
    for(var i=0;i<self.neurons.length;i++) {
      if(dropout_rate>0) {
        if(self.layer_prev!=null&&self.layer_next!=null&&Math.random()<dropout_rate) {
          //dropout this neuron
          self.neurons[i].dropout=true;
          continue;
        }
        self.neurons[i].dropout=false;
      }
      if(self.layer_next==null)
       results[i]=self.neurons[i].evaluate_output(inputs); //Output layer
      else
       results[i]=self.neurons[i].evaluate(inputs);
    }
    //console.log('RESULT layer', self.index, results);
    //If not the output layer, feedforward
    if(self.layer_next) {
      return self.layer_next.evaluate(results, dropout_rate);
    }
    return results;
  }
  /**
   * Calculate the error for the specified neuron on the previous layer
   * for backpropagation
   */
  this.get_error_for=function(previous_layer_neuron_index) {
    var output=0.0;
    
    for(var i=0;i<self.neurons.length;i++) {
      output+=self.neurons[i].weights[previous_layer_neuron_index+1]*self.neurons[i].error;
    }
    
    return output;
  }
  
  this.backpropagate=function(inputs, expected, learning_rate) {
    //console.log('backpropagate', this.index);
    if(self.layer_prev==null) {
      //If input layer, evaluate to initialize values
      self.evaluate(inputs);//, 0.5);
    }
    
    if(self.layer_next!=null) {
      //If not output layer, backpropagate to the end
      self.layer_next.backpropagate(null, expected, learning_rate);
      
      //At this point the backpropagate function works backwards since the next layer 
      //has been checked above
      
      //Step through each neuron calculating its error based on the next layer's error
      for(var i=0;i<self.neurons.length;i++) {
        //Dropout this neuron for this iteration
        if(self.neurons[i].dropout==true)
          continue;
        var e=self.layer_next.get_error_for(i);
        //window.console.log('HiddenNeuron',  self.neurons[i].last_guessed, expected[i], e);
        self.neurons[i].backpropagate(e);
      }
    } else {
      //Output layer, compare the calculated output with the expected output
      var total_error=0.0;
      for(var i=0;i<self.neurons.length;i++) {
        var e=self.neurons[i].calculate_delta(expected[i]);
        self.neurons[i].backpropagate(e);
        
        var lg=self.neurons[i].last_guessed;
        
        if(lg<0)
          lg=0;
        else
          lg=1;
        if(lg!=expected[i])
          total_error++;
        /*
        window.console.log('OutputNeuron',  
                           lg, 
                           expected[i], 
                           self.neurons[i].error,
                           (lg!=expected[i])?"*****":"");*/
        //total_error*=self.neurons[i].error;
      }
      //console.log('LearningRate1: ',learning_rate,(total_error/self.neurons.length));
      
      //Modify learning rate for this run based on the current error rate
      learning_rate*=(1.0+(total_error/self.neurons.length));
      if(learning_rate>1)
        learning_rate=1;
      console.log('LearningRate2: ',learning_rate);
       // window.console.log('OutputErr',  total_error, self.neurons.length, (total_error/self.neurons.length));
    }
    //Back at the input layer, update the weights
    if(self.layer_prev==null) {
      self.backpropagate_update_weights(inputs, learning_rate);
    }
  }
  this.backpropagate_update_weights=function(inputs, learning_rate) {
    var results=new Float32Array(self.neurons.length);//buffer);
    
    for(var i=0;i<self.neurons.length;i++) {
      results[i]=self.neurons[i].backpropagate_update_weights(inputs, learning_rate);
    }
    
    if(self.layer_next!=null) {
      self.layer_next.backpropagate_update_weights(results, learning_rate);
    }
  }
}
//Perceptron static methods
  //Generate random weights for the given input/neuron length combination
  Perceptron.GenerateWeights=function(input_count, neuron_count) {
    window.console.log("Generate NC[",neuron_count,"] with IC[",input_count,"]");
    //Allocate array buffer
    //var buffer = new ArrayBuffer(neuron_count*(input_count+1)*32);
    var results=new Float32Array(neuron_count*(input_count+1));//buffer);
    
    for(var i=0;i<results.length;i++) {
      results[i]=Math.random()*2.0-1.0;
    }
    console.log('RANDOM WEIGHTS: ', results);
    return results;
  }
  //Generate a random Perceptron with the given details
  Perceptron.Random=function(input_count, neuron_count, layer_prev) {
    var weights = Perceptron.GenerateWeights(input_count, neuron_count);
    var p=new Perceptron(input_count, neuron_count, weights, layer_prev);
    
    return p;
  }
  
function test(perceptron, input, expected, return_value_only) {
  var result = perceptron.evaluateb(input);
  var accuracy=0;
  for(var i=0;i<result.length;i++) {
   if(result[i]==expected[i])
     accuracy++;
  }
  accuracy=accuracy/result.length;
  
  if(return_value_only!=true)
   console.log('Eval[', input, ']:',result, 'Exp', expected, Math.round(accuracy*100)+'%');
  
  return accuracy;
}
window.console.log('Run...!');


function generate_AND() {
  var p = Perceptron.Random(2, 10);
  p.add_next_layer(10);
  p.add_next_layer(1);
  return p;
}
function test_AND(p) {
  test(p,[0,0],[0]);
  test(p,[0,1],[0]);
  test(p,[1,0],[0]);
  test(p,[1,1],[1]);
}
function train_AND(p, iterations, learning_rate) {
    for(var i2=0;i2<iterations;i2++) {
     p.backpropagate([0,0], [0], learning_rate);
     p.backpropagate([0,1], [0], learning_rate);
     p.backpropagate([1,0], [0], learning_rate);
     p.backpropagate([1,1], [1], learning_rate);
    }
}
function generate_4bit_adder() {
  var p = Perceptron.Random(4, 100);
  //p.add_next_layer(100);
  p.add_next_layer(3);
  
  return p;
}
function test_4bit_adder(p, show_total_only) {
  var accuracy=0;
  /*
  accuracy+=test(p,[0,0,0,0],[0,0,0,0,0,0], show_total_only);
  accuracy+=test(p,[0,0,0,1],[0,0,0,0,0,1], show_total_only);
  accuracy+=test(p,[0,0,1,0],[0,0,0,0,1,0], show_total_only);
  accuracy+=test(p,[0,0,1,1],[0,0,0,1,0,0], show_total_only);
  
  accuracy+=test(p,[0,1, 0,0],[0,0,0,0,0,1], show_total_only);
  accuracy+=test(p,[0,1, 0,1],[0,0,0,0,1,0], show_total_only);
  accuracy+=test(p,[0,1, 1,0],[0,0,0,1,0,0], show_total_only);
  accuracy+=test(p,[0,1, 1,1],[0,0,1,0,0,0], show_total_only);
  
  accuracy+=test(p,[1,0,0,0],[0,0,0,0,1,0], show_total_only);
  accuracy+=test(p,[1,0,0,1],[0,0,0,1,0,0], show_total_only);
  accuracy+=test(p,[1,0,1,0],[0,0,1,0,0,0], show_total_only);
  accuracy+=test(p,[1,0,1,1],[0,1,0,0,0,0], show_total_only);
  
  accuracy+=test(p,[1,1,0,0],[0,0,0,1,0,0], show_total_only);
  accuracy+=test(p,[1,1,0,1],[0,0,1,0,0,0], show_total_only);
  accuracy+=test(p,[1,1,1,0],[0,1,0,0,0,0], show_total_only);
  accuracy+=test(p,[1,1,1,1],[1,0,0,0,0,0], show_total_only);
  */
  accuracy+=test(p,[0,0,0,0],[0,0,0], show_total_only);
  accuracy+=test(p,[0,0,0,1],[0,0,1], show_total_only);
  accuracy+=test(p,[0,0,1,0],[0,1,0], show_total_only);
  accuracy+=test(p,[0,0,1,1],[0,1,1], show_total_only);
  
  accuracy+=test(p,[0,1, 0,0],[0,0,1], show_total_only);
  accuracy+=test(p,[0,1, 0,1],[0,1,0], show_total_only);
  accuracy+=test(p,[0,1, 1,0],[0,1,1], show_total_only);
  accuracy+=test(p,[0,1, 1,1],[1,0,0], show_total_only);
  
  accuracy+=test(p,[1,0,0,0],[0,1,0], show_total_only);
  accuracy+=test(p,[1,0,0,1],[0,1,1], show_total_only);
  accuracy+=test(p,[1,0,1,0],[1,0,0], show_total_only);
  accuracy+=test(p,[1,0,1,1],[1,0,1], show_total_only);
  
  accuracy+=test(p,[1,1,0,0],[0,1,1], show_total_only);
  accuracy+=test(p,[1,1,0,1],[1,0,0], show_total_only);
  accuracy+=test(p,[1,1,1,0],[1,0,1], show_total_only);
  accuracy+=test(p,[1,1,1,1],[1,1,0], show_total_only);
  
  accuracy=accuracy/16;
  window.console.log('Total Accuracy: ', (accuracy*100)+"%");
}
function train_4bit_adder(p, iterations, learning_rate) {
  /*
  //Break into chunks of 10
  var st_chunks=10;
  var loops=Math.round(iterations/st_chunks);
  
  if(st_chunks<=0||loops<=0) {
    st_chunks=iterations;
    loops=1;
  }
  console.log('Train ',loops, st_chunks, st_chunks*loops);
  */
  //for(var i=0;i<loops;i++) {
  //setTimeout(function() {
  //  for(var i2=0;i2<st_chunks;i2++) {
  for(var i=0;i<iterations;i++) {
    /*
      p.backpropagate([0,0, 0,0], [0,0,0,0,0,0], learning_rate);
      p.backpropagate([0,0, 0,1], [0,0,0,0,0,1], learning_rate);
      p.backpropagate([0,0, 1,0], [0,0,0,0,1,0], learning_rate);
      p.backpropagate([0,0, 1,1], [0,0,0,1,0,0], learning_rate);

      p.backpropagate([0,1, 0,0], [0,0,0,0,0,1], learning_rate);
      p.backpropagate([0,1, 0,1], [0,0,0,0,1,0], learning_rate);
      p.backpropagate([0,1, 1,0], [0,0,0,1,0,0], learning_rate);
      p.backpropagate([0,1, 1,1], [0,0,1,0,0,0], learning_rate);

      p.backpropagate([1,0, 0,0], [0,0,0,0,1,0], learning_rate);
      p.backpropagate([1,0, 0,1], [0,0,0,1,0,0], learning_rate);
      p.backpropagate([1,0, 1,0], [0,0,1,0,0,0], learning_rate);
      p.backpropagate([1,0, 1,1], [0,1,0,0,0,0], learning_rate);

      p.backpropagate([1,1, 0,0], [0,0,0,1,0,0], learning_rate);
      p.backpropagate([1,1, 0,1], [0,0,1,0,0,0], learning_rate);
      p.backpropagate([1,1, 1,0], [0,1,0,0,0,0], learning_rate);
      p.backpropagate([1,1, 1,1], [1,0,0,0,0,0], learning_rate);
    */
      p.backpropagate([0,0, 0,0], [0,0,0], learning_rate);
      p.backpropagate([0,0, 0,1], [0,0,1], learning_rate);
      p.backpropagate([0,0, 1,0], [0,1,0], learning_rate);
      p.backpropagate([0,0, 1,1], [0,1,1], learning_rate);

      p.backpropagate([0,1, 0,0], [0,0,1], learning_rate);
      p.backpropagate([0,1, 0,1], [0,1,0], learning_rate);
      p.backpropagate([0,1, 1,0], [0,1,1], learning_rate);
      p.backpropagate([0,1, 1,1], [1,0,0], learning_rate);

      p.backpropagate([1,0, 0,0], [0,1,0], learning_rate);
      p.backpropagate([1,0, 0,1], [0,1,1], learning_rate);
      p.backpropagate([1,0, 1,0], [1,0,0], learning_rate);
      p.backpropagate([1,0, 1,1], [1,0,1], learning_rate);

      p.backpropagate([1,1, 0,0], [0,1,1], learning_rate);
      p.backpropagate([1,1, 0,1], [1,0,0], learning_rate);
      p.backpropagate([1,1, 1,0], [1,0,1], learning_rate);
      p.backpropagate([1,1, 1,1], [1,1,0], learning_rate);
  //  }
  //}, 1000);
  }
}

var lr=0.05;
/*
var pand=generate_AND();
  test_AND(pand);
  train_AND(pand, 2000, lr);

  window.console.log('************************************');
  test_AND(pand);

  window.console.log('************************************');
  window.console.log('************************************');
  window.console.log('************************************');*/
var p4bit_adder=generate_4bit_adder();
  test_4bit_adder(p4bit_adder);
  for(var i=0;i<2;i++) {
   train_4bit_adder(p4bit_adder, 2, lr);
   test_4bit_adder(p4bit_adder, true);
  }
  window.console.log('************************************');
  test_4bit_adder(p4bit_adder);
/*
var p4bit_adder=generate_4bit_adder();

var lr=0.05;
for(var i=0;i<2;i++) {
  //console.log('Eval[', i, ']: ',p.evaluateb([0,1,0,1]));
  test_4bit_adder(generate_4bit_adder);
  window.console.log('************************************');
  train_4bit_adder(generate_4bit_adder, 200, lr);
}
  test_4bit_adder(generate_4bit_adder);
*/
window.console.log('DONE!');
