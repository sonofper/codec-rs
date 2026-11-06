Steps to build Huffman Tree
This algorithm builds a tree in bottom up manner using a priority queue (or
heap). Input is an array of unique characters along with their frequency of
occurrences and output is Huffman Tree.

1. Create a leaf node for each unique character and build a min heap of all leaf
   nodes (Min Heap is used as a priority queue. The value of frequency field is
   used to compare two nodes in min heap. Initially, the least frequent character
   is at root)

2. Extract two nodes with the minimum frequency from the min heap.

3. Create a new internal node with a frequency equal to the sum of the two nodes
   frequencies. Make the first extracted node as its left child and the other
   extracted node as its right child. Add this node to the min heap.

4. Repeat steps#2 and #3 until the heap contains only one node. The remaining
   node is the root node and the tree is complete.
